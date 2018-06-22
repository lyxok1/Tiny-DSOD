#include <vector>
#include "caffe/layers/correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void CorrelationForward(const int nthreads,
    const Dtype* b0_data, const Dtype* b1_data, const int num, const int channels,
    const int b0_height, const int b0_width, const int b1_height, const int b1_width,
    const int displacement_, const int dilation_, const Dtype step_h, const Dtype step_w,
    const int b0_dim, const int b0_spatial, const int b1_dim, const int b1_spatial,
    const int top_dim, const int top_spatial, const int top_width, Dtype* const top_data) {

    CUDA_KERNEL_LOOP(index, nthreads)
    {
      const int n = index/top_dim;
      const int out_c = (index%top_dim)/top_spatial;
      const int y = (index%top_spatial)/top_width;
      const int x = index%top_width;

      const int window_size = displacement_*2+1;
      const int b0_x = x*step_w;
      const int b0_y = y*step_h;
      const int dx = (out_c%window_size) - displacement_;
      const int dy = out_c/window_size - displacement_;

      Dtype temp = 0.0;
      const int offset_x = b0_x + dilation_*dx;
      const int offset_y = b0_y + dilation_*dy;
      if(offset_x>=0 && offset_x < b0_width && offset_y>=0 && offset_y<b0_height)
      {
        for(int c=0; c<channels; c++)
          temp += b1_data[n*b1_dim+c*b1_spatial+y*b1_width+x]
                *b0_data[n*b0_dim+c*b0_spatial+offset_y*b0_width+offset_x];
      }
      top_data[index] = temp/channels;
    }
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if(self_)
  {
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int bottom_height = bottom[0]->height();
    const int bottom_width = bottom[0]->width();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int b0_dim = bottom[0]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int top_width = bottom_width;
    const int top_spatial = b0_spatial;
    const int window_size = displacement_*2+1;
    const int top_dim = top_spatial*window_size*window_size;
    const int count = top[0]->count();

    CorrelationForward<Dtype>
          // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_data, num, channels,
        bottom_height, bottom_width, bottom_height, bottom_width,
        displacement_, dilation_, step_h_, step_w_,
        b0_dim, b0_spatial, b0_dim, b0_spatial, top_dim, top_spatial, top_width, top_data);
  }
  else
  {
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int bottom0_height = bottom[0]->height();
    const int bottom0_width = bottom[0]->width();
    const int bottom1_height = bottom[1]->height();
    const int bottom1_width = bottom[1]->width();
    const Dtype* bottom0_data = bottom[0]->gpu_data();
    const Dtype* bottom1_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int b0_dim = bottom[0]->count(1);
    const int b1_dim = bottom[1]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int b1_spatial = bottom[1]->count(2);
    const int top_width = bottom1_width;
    const int top_spatial = b1_spatial;
    const int window_size = displacement_*2+1;
    const int top_dim = top_spatial*window_size*window_size;
    const int count = top[0]->count();

    CorrelationForward<Dtype>
          // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom0_data, bottom1_data, num, channels,
        bottom0_height, bottom0_width, bottom1_height, bottom1_width,
        displacement_, dilation_, step_h_, step_w_,
        b0_dim, b0_spatial, b1_dim, b1_spatial, top_dim, top_spatial, top_width, top_data);
  }
}

template <typename Dtype>
__global__ void SelfCorrelationBackward(const int nthreads, 
      const Dtype* top_diff, const Dtype* b0_data, const int num,
      const int channels, const int b0_height, const int b0_width,
      const int displacement_, const int dilation_, const int b0_dim, const int b0_spatial,
      const int top_dim, const int top_spatial, const int top_width, Dtype* b0_diff) {
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const int n = index / b0_dim;
    const int c = (index % b0_dim) / b0_spatial;
    const int y = (index % b0_spatial) / b0_width;
    const int x = index % b0_width;
    
    const int window_size = displacement_*2+1;
    
    Dtype temp = 0.0;

    for(int dx=-1*displacement_; dx<=displacement_; dx++)
      for(int dy=-1*displacement_; dy<=displacement_; dy++)
      {
        if(dx==0&&dy==0)
        { 
          int out_c = (displacement_+dy)*window_size+(displacement_+dx);
          temp += 2*b0_data[index]
                  *top_diff[n*top_dim+out_c*top_spatial+y*top_width+x];
        }
        else
        {
          const int offset_x = x + dx*dilation_;
          const int offset_y = y + dy*dilation_;
          if(offset_x >=0 && offset_x<b0_width && offset_y>=0 && offset_y<b0_height)
          {
              int out_c = (displacement_ + dy)*window_size+(displacement_ + dx);
              temp += b0_data[n*b0_dim+c*b0_spatial+offset_y*b0_width+offset_x]
                    *top_diff[n*top_dim+out_c*top_spatial+y*top_width+x];

              out_c = (displacement_ - dy)*window_size+(displacement_ - dx);
              temp += b0_data[n*b0_dim+c*b0_spatial+offset_y*b0_width+offset_x]
                    *top_diff[n*top_dim+out_c*top_spatial+offset_y*top_width+offset_x];
          }
        }
      }
    b0_diff[index] = temp/channels;
  }
}

template <typename Dtype>
__global__ void CorrelationBackward0(const int nthreads, 
      const Dtype* top_diff, const Dtype* b1_data, const int num,
      const int channels, const int b0_height, const int b0_width, 
      const int b1_height,const int b1_width,
      const int displacement_, const int dilation_,
      const Dtype step_h, const Dtype step_w, const int b0_dim, const int b0_spatial,
      const int b1_dim, const int b1_spatial, const int top_dim, const int top_spatial,
      const int top_width, Dtype* b0_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / b0_dim;
    const int c = (index % b0_dim) / b0_spatial;
    const int y = (index % b0_spatial) / b0_width;
    const int x = index % b0_width;
    const int window_size = displacement_*2+1;
    
    Dtype temp = 0.0;
    
    for(int dx=-1*displacement_; dx<=displacement_; dx++)
      for(int dy=-1*displacement_; dy<=displacement_; dy++)
      {
        const int b0_offset_x = x + dx*dilation_;
        const int b0_offset_y = y + dy*dilation_;
        if(b0_offset_x >=0 && b0_offset_x<b0_width && b0_offset_y>=0 && b0_offset_y<b0_height)
        {
          // find the pixels of d1 mapped to the current position
          const int candidate_x = int(x/step_w);
          const int candidate_y = int(y/step_h);
          const int max_candidate_x = candidate_x + int(1.0/step_w) + 1;
          const int max_candidate_y = candidate_y + int(1.0/step_h) + 1;
          const int out_c = (displacement_ - dy)*window_size+(displacement_ - dx);

          for(int i = candidate_x; i<=max_candidate_x; i++)
            for(int j = candidate_y; j<=max_candidate_y; j++)
            {
              if(int(i*step_w)==b0_offset_x && int(j*step_h)==b0_offset_y)
              {
                temp += top_diff[n*top_dim+out_c*top_spatial+j*top_width+i]
                        *b1_data[n*b1_dim+c*b1_spatial+j*b1_width+i];
              }
            }
        }
      }
    b0_diff[index] = temp/channels;
  }
}

template <typename Dtype>
__global__ void CorrelationBackward1(const int nthreads, 
      const Dtype* top_diff, const Dtype* b0_data, const int num,
      const int channels, const int b0_height, const int b0_width, 
      const int b1_height,const int b1_width,
      const int displacement_, const int dilation_,
      const Dtype step_h, const Dtype step_w, const int b0_dim, const int b0_spatial,
      const int b1_dim, const int b1_spatial, const int top_dim, const int top_spatial,
      const int top_width, Dtype* b1_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / b1_dim;
    const int c = (index % b1_dim) / b1_spatial;
    const int y = (index % b1_spatial) / b1_width;
    const int x = index % b1_width;
    
    const int window_size = displacement_*2+1;  

    Dtype temp = 0.0;
    
    const int b0_x = x*step_w;
    const int b0_y = y*step_h;
    for(int dx=-1*displacement_; dx<=displacement_; dx++)
      for(int dy=-1*displacement_; dy<=displacement_; dy++)
      {
        const int offset_x = b0_x + dilation_*dx;
        const int offset_y = b0_y + dilation_*dy;
        const int out_c = (dy + displacement_)*window_size+(dx + displacement_);

        if(offset_x < b0_width && offset_x >=0 && offset_y < b0_height && offset_y >= 0)
        {
            temp += b0_data[n*b0_dim+c*b0_spatial+offset_y*b0_width+offset_x]
                    *top_diff[n*top_dim+out_c*top_spatial+y*top_width+x];
        }
      }
    b1_diff[index] = temp/channels;
  }
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Backward_gpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  if(self_)
  {
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int bottom_height = bottom[0]->height();
    const int bottom_width = bottom[0]->width();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int b0_dim = bottom[0]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int window_size = 2*displacement_+1;
    const int top_width = bottom_width;
    const int top_spatial = b0_spatial;
    const int top_dim = top_spatial*window_size*window_size;

    if(propagate_down[0])
    {
      const int count = bottom[0]->count();
      SelfCorrelationBackward<Dtype>
      <<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, num, channels, bottom_height, bottom_width,
        displacement_, dilation_, b0_dim, b0_spatial, top_dim, top_spatial, top_width, bottom_diff);
    }
  }
  else
  {
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int bottom0_height = bottom[0]->height();
    const int bottom0_width = bottom[0]->width();
    const int bottom1_height = bottom[1]->height();
    const int bottom1_width = bottom[1]->width();
    const Dtype* bottom0_data = bottom[0]->gpu_data();
    const Dtype* bottom1_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom0_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom1_diff = bottom[1]->mutable_gpu_diff();
    const int b0_dim = bottom[0]->count(1);
    const int b1_dim = bottom[1]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int b1_spatial = bottom[1]->count(2);
    const int window_size = 2*displacement_+1;
    const int top_width = bottom1_width;
    const int top_spatial = b1_spatial;
    const int top_dim = top_spatial*window_size*window_size;
    
    if (propagate_down[1]) {
      const int b1_count = bottom[1]->count();
      CorrelationBackward1<Dtype>
            // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(b1_count), CAFFE_CUDA_NUM_THREADS>>>(
        b1_count, top_diff, bottom0_data, num, channels,
        bottom0_height, bottom0_width, bottom1_height, bottom1_width,
        displacement_, dilation_, step_h_, step_w_,
        b0_dim, b0_spatial, b1_dim, b1_spatial, top_dim, top_spatial, top_width, bottom1_diff);
    }
    if (propagate_down[0]) {
      const int b0_count = bottom[0]->count();
      CorrelationBackward0<Dtype>
            // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(b0_count), CAFFE_CUDA_NUM_THREADS>>>(
          b0_count, top_diff, bottom1_data, num, channels,
          bottom0_height, bottom0_width, bottom1_height, bottom1_width,
          displacement_, dilation_, step_h_, step_w_,
          b0_dim, b0_spatial, b1_dim, b1_spatial, top_dim, top_spatial, top_width, bottom0_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CorrelationLayer);

}  // namespace caffe