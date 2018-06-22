#include <vector>

#include "caffe/layers/norm_conv_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void NormConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void NormConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK(!(this->layer_param_.convolution_param().bias_term())) << "norm convolution does not support bias";
  LayerParameter layer_param;
  layer_param.set_type("Normalize");
  layer_param.set_name("weight_norm");
  layer_param.mutable_norm_param()->mutable_scale_filler()->set_type("constant");
  layer_param.mutable_norm_param()->mutable_scale_filler()->set_value(1.0);
  norm_layer_.reset(new NormalizeLayer<Dtype>(layer_param));

  norm_bottom_vec_.push_back(this->blobs_[0].get());
  norm_top_vec_.push_back(&norm_weight_);
  norm_layer_->SetUp(norm_bottom_vec_, norm_top_vec_);

}

template <typename Dtype>
void NormConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  norm_layer_->Reshape(norm_bottom_vec_, norm_top_vec_);

}

template <typename Dtype>
void NormConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Normalize the inside weight first
  norm_layer_->Forward(norm_bottom_vec_, norm_top_vec_);
  const Dtype* weight = norm_weight_.cpu_data();
    vector<int> shape = this->blobs_[0]->shape();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
        this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                                      top_data + n * this->top_dim_);
    }
  }
}

template <typename Dtype>
void NormConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = norm_weight_.cpu_data();
  Dtype* weight_diff = norm_weight_.mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
      // backward gradient from norm weight to weight
      norm_layer_->Backward(norm_top_vec_, propagate_down, norm_bottom_vec_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(NormConvolutionLayer);
#endif

INSTANTIATE_CLASS(NormConvolutionLayer);
REGISTER_LAYER_CLASS(NormConvolution);

}  // namespace caffe
