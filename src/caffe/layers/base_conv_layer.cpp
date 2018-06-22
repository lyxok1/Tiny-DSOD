#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include <time.h>

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  using_approximate_ = conv_param.using_approximate();
  opt_ = conv_param.optimization();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }

  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  if(opt_==ConvolutionParameter_Optimization_KN2ROW)
  {
    CHECK_EQ(kernel_shape_data[0]%2,1) << "in kn2row optimization, the kernel size should be odd";
    CHECK_EQ(kernel_shape_data[1]%2,1) << "in kn2row optimization, the kernel size should be odd";
    CHECK_EQ(stride_data[0],1);
    CHECK_EQ(stride_data[1],1);
    CHECK_EQ(pad_data[0],kernel_shape_data[0]/2);
    CHECK_EQ(pad_data[1],kernel_shape_data[1]/2);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }

  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();

  vector<int> top_shape(bottom[0]->shape().begin(),
        bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_); // add 2*delta channels at the output for shift-add operation
  for (int i = 0; i < num_spatial_axes_; ++i) {
      top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
      top[top_id]->Reshape(top_shape);
  }

  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }

  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }

  if(opt_==ConvolutionParameter_Optimization_IM2COL)
  {
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
    col_buffer_shape_.clear();
    col_buffer_shape_.push_back(kernel_dim_ * group_);

    for (int i = 0; i < num_spatial_axes_; ++i) {
      if (reverse_dimensions()) {
        col_buffer_shape_.push_back(input_shape(i + 1));
      } else {
        col_buffer_shape_.push_back(output_shape_[i]);
      }
    }
    col_buffer_.Reshape(col_buffer_shape_);
  }


  //int ih = input_shape(1);
  //int kw = this->blobs_[0]->shape(3);
  //int ic = input_shape(0);
  //int ow = output_shape_[1];
//  LOG(INFO) << " num " << col_buffer_shape_[1]\
//            << " kh " << this->blobs_[0]->shape(2)\
//            << " kw " << this->blobs_[0]->shape(3)\
//            << " kc " << this->blobs_[0]->shape(1)\
//            << " kkc " << col_buffer_shape_[0]\
//            << " ic " << input_shape(0)\
//            << " ih " << input_shape(1)\
//            << " iw " << input_shape(2)\
//            << " oh " << output_shape_[0]\
//            << " ow " << output_shape_[1];
 if(opt_==ConvolutionParameter_Optimization_KN2ROW)
  {
    const int* kernel_shape_data = kernel_shape_.cpu_data();
    col_buffer_kernel_shape_.clear();

    col_buffer_kernel_shape_.push_back(conv_in_channels_);
    col_buffer_kernel_shape_.push_back(conv_out_channels_);

    weights_buffer_.Reshape(col_buffer_kernel_shape_);

    int delta = 0;
    delta = static_cast<int>(kernel_shape_data[0]/(2*(bottom[0]->height()))) + 1;
    
    output_buffer_shape_.clear();
    int output_shape = output_shape_[0]*output_shape_[1];
    int num_filter_shape = this->blobs_[0]->shape(0);
    output_buffer_shape_.push_back(output_shape);
    output_buffer_shape_.push_back(num_filter_shape + 2*delta);
    output_buffer_.Reshape(output_buffer_shape_);

    if(!using_approximate_)
    {
      output_buffer_shape_[1] -= 2*delta;
      tune_buffer_.Reshape(output_buffer_shape_);
    }
  }

  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  //clock_t startTime, endTime;
  //startTime =clock();
  if(opt_==ConvolutionParameter_Optimization_IM2COL)
  {
      if (!is_1x1_) {
        if (!skip_im2col) {
          conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
        }
        col_buff = col_buffer_.cpu_data();
      }
      for (int g = 0; g < group_; ++g) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
            group_, conv_out_spatial_dim_, kernel_dim_,
            (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
            (Dtype)0., output + output_offset_ * g);

      }
  }
  else if(opt_==ConvolutionParameter_Optimization_KN2ROW)
  {
      const int input_channels = this->blobs_[0]->shape(channel_axis_);
      const int kernel_h = this->blobs_[0]->shape(channel_axis_+1);
      const int kernel_w = this->blobs_[0]->shape(channel_axis_+2);
      const int output_width = output_shape_[0];
      const int channel_size = kernel_h*kernel_w;

      //kn2row_cpu(weights, input_channels,
      //           kernel_h, kernel_w, num_output_,
      //          weights_buffer_.mutable_cpu_data());

      Dtype* output_buffer = output_buffer_.mutable_cpu_data();

      int output_h = output_shape_[0];
      int output_w = output_shape_[1];
      int output_spatial = output_w*output_h;
      int delta = 0;
      delta = static_cast<int>(kernel_h/(2*(output_h))) + 1;
      output_buffer += delta*output_spatial;
      //caffe_set(output_spatial*num_output_,Dtype(0),output_buffer);

      if(using_approximate_)
        for(int y=0; y < kernel_h ; y++)
          for(int x=0; x< kernel_w ; x++)
          {
            kn2row_cpu(weights, x, y, input_channels,
                       kernel_h, kernel_w, num_output_,
                       weights_buffer_.mutable_cpu_data());
            if(x==0&&y==0)
              caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                                  output_spatial, input_channels,
                                  (Dtype) 1., weights_buffer_.cpu_data(), input,
                                  (Dtype) 0., output_buffer - (y-kernel_h/2)*output_w - (x - kernel_w/2));
            else
              caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                                  output_spatial, input_channels,
                                  (Dtype) 1., weights_buffer_.cpu_data(), input,
                                  (Dtype) 1., output_buffer - (y-kernel_h/2)*output_w - (x - kernel_w/2));
          }
      else
      {
	caffe_set(output_buffer_.count(),Dtype(0),output_buffer_.mutable_cpu_data());
        Dtype* tune_buffer = tune_buffer_.mutable_cpu_data();
        for(int y=0; y < kernel_h ; y++)
          for(int x=0; x< kernel_w ; x++)
          {
            kn2row_cpu(weights, x, y, input_channels,
                       kernel_h, kernel_w, num_output_,
                       weights_buffer_.mutable_cpu_data());

            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                                  output_spatial, input_channels,
                                  (Dtype) 1., weights_buffer_.cpu_data(), input,
                                  (Dtype) 0., tune_buffer);
            if(x<kernel_w/2)
              for(int m=0;m<num_output_;m++)
                for(int i=0;i<kernel_w/2-x;i++)
                  for(int j=0;j<output_h;j++)
                    tune_buffer[m*output_spatial+j*output_w+output_w-i-1] = 0;

            if(x>kernel_w/2)
              for(int m=0;m<num_output_;m++)
                for(int i=0;i<x-kernel_w/2;i++)
                  for(int j=0;j<output_h;j++)
                  tune_buffer[m*output_spatial+j*output_w+i] = 0;

            caffe_cpu_axpby(output_spatial*num_output_, Dtype(1.), tune_buffer_.cpu_data(),
                            Dtype(1.), output_buffer - (y-kernel_h/2)*output_w - (x - kernel_w/2));

          }
      }

      caffe_copy(output_spatial*num_output_,output_buffer,output);
  }
  else
    LOG(FATAL) << "the optimization type does not exist";

    //endTime = clock();
    //LOG(INFO) << "gemm cost:" << ((endTime - startTime)/(double)CLOCKS_PER_SEC) << "s";
}

/*
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm_kernel(const Dtype *input,
                                                          const Dtype *weights, Dtype *output, bool skip_im2col) {
    const Dtype* col_buff = input;

    Dtype *weights_buffer;
    Dtype *output_buffer;

    if (!is_1x1_ && conv_out_spatial_dim_!=1) {
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                weights_buffer = weights_buffer_.mutable_cpu_data();
                output_buffer = output_buffer_.mutable_cpu_data() + conv_out_spatial_dim_ - output_shape_[0] -1;

                for (int n = 0; n < conv_out_channels_; ++n) {
                    for (int c = 0; c < conv_in_channels_; ++c) {
                        *(weights_buffer++) = weights[n * filter_size + c * channel_size + i * kernel_w + j];
                    }
                }
                weights_buffer = weights_buffer_.mutable_cpu_data();
                if (i == 0 and j == 0) {
                    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
                                          conv_out_spatial_dim_, conv_in_channels_,
                                          (Dtype) 1., weights_buffer, col_buff,
                                          (Dtype) 0., output_buffer + 2 * output_shape_[0] + 2);
                } else {
                    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
                                          conv_out_spatial_dim_, conv_in_channels_,
                                          (Dtype) 1., weights_buffer, col_buff,
                                          (Dtype) 1., output_buffer + (2 - i) * output_shape_[0] + 2 - j);
                }
            }
        }
        output_buffer = output_buffer_.mutable_cpu_data();
        int output_size = conv_out_spatial_dim_ * conv_out_channels_;
        output_buffer += conv_out_spatial_dim_;

        for (int i = 0; i < output_size; i++) {
            *(output + i) = *(output_buffer + i);
        }
    }
    else{
        for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
                                                              group_, conv_out_spatial_dim_, kernel_dim_,
                                  (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
                                  (Dtype)0., output + output_offset_ * g);
        }
    }
    endTime = clock();
    LOG(INFO) << "gemm cost:" << ((endTime - startTime)/(double)CLOCKS_PER_SEC) << "s";

//  if (!is_1x1_) {
//    if (!skip_im2col) {
//      conv_im2col_cpu_mec(input, col_buffer_.mutable_cpu_data());
//    }
//    col_buff = col_buffer_.cpu_data();
//  }
//  for (int g = 0; g < group_; ++g) {
//    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
//                                                      group_, conv_out_spatial_dim_, kernel_dim_,
//                          (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
//                          (Dtype)0., output + output_offset_ * g);
//  }
}
*/

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  if(opt_==ConvolutionParameter_Optimization_IM2COL)
  {
      Dtype* col_buff = col_buffer_.mutable_gpu_data();
      if (is_1x1_) {
        col_buff = input;
      }
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
            conv_out_spatial_dim_, conv_out_channels_ / group_,
            (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
            (Dtype)0., col_buff + col_offset_ * g);
      }
      if (!is_1x1_) {
        conv_col2im_gpu(col_buff, input);
      }
  }
  else
  {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
