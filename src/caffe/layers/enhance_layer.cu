#include <vector>

#include "caffe/layers/enhance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EnhanceForward(const int nthread, const Dtype* bottom_data, const Dtype* coeff, const int spatial, 
  const int width, const int height, const int channels, const int data_dim , const int coeff_dim,
  const int displacement_, const int dilation_, Dtype* top_data) {
  
  CUDA_KERNEL_LOOP(index, nthread)
  {
    const int n = index/(data_dim);
    const int c = (index%data_dim)/spatial;
    const int y = (index%spatial)/width;
    const int x = index%width;

    const int window = 2*displacement_+1;

    Dtype temp = 0.0;
    for(int dx=-1*displacement_;dx<=displacement_;dx++)
      for(int dy=-1*displacement_;dy<=displacement_;dy++)
      {
        const int offset_x = x + dx*dilation_;
        const int offset_y = y + dy*dilation_;
        const int out_c = (dy+displacement_)*window+(dx+displacement_);

        if(offset_x>=0 && offset_x<width && offset_y>=0 && offset_y<height)
        {
          temp += coeff[n*coeff_dim+out_c*spatial+y*width+x]
          *bottom_data[n*data_dim+c*spatial+offset_y*width+offset_x];
        }
      }
    top_data[index] = temp;
  }
}

template <typename Dtype>
void EnhanceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // obtain correlation parameters
  correlation_->Forward(bottom, corr_top_vec_);
  softmax_->Forward(corr_top_vec_, soft_top_vec_);

  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* coeff_data = soft_.gpu_data();
  const int width = top[0]->width();
  const int height = top[0]->height();
  const int channels = top[0]->channels();
  const int blocks = top[0]->count();
  const int data_dim = channels*width*height;
  const int data_spatial = width*height;
  const int coeff_dim = soft_.count(1);

  caffe_gpu_set(top[0]->count(),Dtype(0.), top_data);
  EnhanceForward<Dtype><<<CAFFE_GET_BLOCKS(blocks),CAFFE_CUDA_NUM_THREADS>>>
       (blocks, bottom_data, coeff_data, data_spatial, 
        width, height, channels, data_dim, coeff_dim, displacement_, dilation_, top_data);
}

template <typename Dtype>
__global__ void EnhanceBackwardCoeff(const int nthread, const Dtype* bottom_data, const Dtype* top_diff, const int spatial, const int coeff_dim, 
  const int data_dim, const int width, const int height, const int channels, const int displacement_, const int dilation_, Dtype* coeff_diff) {
  CUDA_KERNEL_LOOP(index, nthread)
  {
    const int n = index/coeff_dim;
    const int out_c = (index%coeff_dim)/spatial;
    const int y = (index%spatial)/width;
    const int x = index%width;
    const int window = 2*displacement_+1;

    const int dx = out_c%window - displacement_;
    const int dy = out_c/window - displacement_;
    const int offset_x = x + dx*dilation_;
    const int offset_y = y + dy*dilation_;

    Dtype temp = 0.0;
    if(offset_x>=0 && offset_x<width && offset_y>=0 && offset_y<height)
      for(int c=0;c<channels;c++)
      {
        temp += top_diff[n*data_dim+c*spatial+y*width+x]
          *bottom_data[n*data_dim+c*spatial+offset_y*width+offset_x];
      }
    coeff_diff[index] = temp;
  }

}

template <typename Dtype>
__global__ void EnhanceBackwardData(const int nthread, const Dtype* coeff_data, const Dtype* top_diff, const int spatial, const int coeff_dim, 
  const int data_dim, const int width, const int height, const int channels, const int displacement_, const int dilation_, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthread)
  {
    const int n = index/data_dim;
    const int c = (index%data_dim)/spatial;
    const int y = (index%spatial)/width;
    const int x = index%width;
    const int window = 2*displacement_+1;

    Dtype temp = 0.0;
    for(int dx=-1*displacement_;dx<=displacement_;dx++)
      for(int dy=-1*displacement_;dy<=displacement_;dy++)
      {
        const int offset_x = x + dx*dilation_;
        const int offset_y = y + dy*dilation_;
        const int out_c = (displacement_ - dy)*window+(displacement_ - dx);
        if(offset_x>=0 && offset_x<width && offset_y>=0 && offset_y<height)
        {
          temp += top_diff[n*data_dim+c*spatial+offset_y*width+offset_x]
            *coeff_data[n*coeff_dim+out_c*spatial+offset_y*width+offset_x];
        }
      }
    bottom_diff[index] += temp;
  }

}

template <typename Dtype>
void EnhanceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* coeff_diff = soft_.mutable_gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* coeff_data = soft_.gpu_data();
  const int width = top[0]->width();
  const int height = top[0]->height();
  const int channels = top[0]->channels();
  const int data_dim = channels*width*height;
  const int data_spatial = width*height;
  const int coeff_dim = soft_.count(1);
  const int data_count = top[0]->count();
  const int coeff_count = soft_.count();

  caffe_gpu_set(data_count, Dtype(0.), bottom_diff);

  if(propagate_down[0])
  {
    //LOG(INFO) << "start backward coeff";
    EnhanceBackwardCoeff<Dtype><<<CAFFE_GET_BLOCKS(coeff_count),CAFFE_CUDA_NUM_THREADS>>>
    (coeff_count, bottom_data, top_diff, data_spatial, coeff_dim, data_dim, width, height, channels,
      displacement_, dilation_, coeff_diff);

    //LOG(INFO) << "backwrad from coeff to bottom";
    softmax_->Backward(soft_top_vec_, propagate_down, corr_top_vec_);
    correlation_->Backward(corr_top_vec_, propagate_down, bottom);

    //LOG(INFO) << "start backward data"
    EnhanceBackwardData<Dtype><<<CAFFE_GET_BLOCKS(data_count),CAFFE_CUDA_NUM_THREADS>>>
    (data_count, coeff_data, top_diff, data_spatial, coeff_dim, 
     data_dim, width, height, channels, displacement_, dilation_, bottom_diff);

    //LOG(INFO) << "finish";
  }
  else
  {// pass gradient directly from top
    caffe_gpu_memcpy(data_count*sizeof(Dtype), top_diff, bottom_diff);
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(EnhanceLayer);

}  // namespace caffe
