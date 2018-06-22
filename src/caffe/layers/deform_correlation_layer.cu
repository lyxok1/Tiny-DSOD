#include <vector>
#include "caffe/layers/deform_correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void DeformCorrelationForward(const int nthreads,
    const Dtype* b0_data, const Dtype* b1_data, const Dtype* offset_data, const int num, const int channels,
    const int b0_height, const int b0_width, const int b1_height, const int b1_width,
    const int displacement, const int dilation, const int step_h, const int step_w,
    const int window_size, const int b0_dim, const int b0_spatial, const int b1_dim, const int b1_spatial,
    const int offset_dim,const int top_dim, const int top_spatial, const int top_width, Dtype* top_data) {
  
    extern __shared__ unsigned char kcache[];
    Dtype* cache = (Dtype*)kcache;

    const int index = blockIdx.x;
    const int c = threadIdx.x;
    const int n = index/top_dim;
    const int out_c = (index%top_dim)/top_spatial;
    const int y = (index%top_spatial)/top_width;
    const int x = index%top_width;

    const int b0_x = x*step_w;
    const int b0_y = y*step_h;
    const int dx = (out_c%window_size) - displacement;
    const int dy = out_c/window_size - displacement;

    const Dtype offset_x = offset_data[n*offset_dim+out_c*2*b1_spatial+y*b1_width+x];
    const Dtype offset_y = offset_data[n*offset_dim+(out_c*2+1)*b1_spatial+y*b1_width+x];
    const Dtype h = b0_y + dy*dilation + offset_y;
    const Dtype w = b0_x + dx*dilation + offset_x;

    //bilinear interpolation
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    Dtype hh = h - h_low;
    Dtype lh = 1 - hh;
    Dtype hw = w - w_low;
    Dtype lw = 1 - hw;
    Dtype temp = 0.0;
    if(h_low >=0 && h_high < b0_height && w_low >= 0 && w_high < b0_width)
    {
      Dtype v1 = b0_data[n*b0_dim+c*b0_spatial+h_high*b0_width+w_high];
      Dtype v2 = b0_data[n*b0_dim+c*b0_spatial+h_low*b0_width+w_high];
      Dtype v3 = b0_data[n*b0_dim+c*b0_spatial+h_high*b0_width+w_low];
      Dtype v4 = b0_data[n*b0_dim+c*b0_spatial+h_low*b0_width+w_low];
      temp = hh*hw*v1 + lh*hw*v2 + hh*lw*v3 + lw*lh*v4;
    }

    temp *= b1_data[n*b1_dim+c*b1_spatial+y*b1_width+x];

    cache[c] = temp/(channels);
    
    __syncthreads();
    int half = blockDim.x/2;
    while(half!=0)
    {
      if(threadIdx.x < half)
        cache[threadIdx.x] += cache[threadIdx.x+half];
      __syncthreads();
      half /= 2;
    }

    if(threadIdx.x==0)
      top_data[index] = cache[threadIdx.x];
}

template <typename Dtype>
void DeformCorrelationLayer<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
    if(!self_)
    {
      const int num = bottom[0]->num();
      const int channels = bottom[0]->channels();
      const int b0_height = bottom[0]->height();
      const int b0_width = bottom[0]->width();
      const int b1_height = bottom[1]->height();
      const int b1_width = bottom[1]->width(); // b1 height width is equal to offset height width
      const Dtype* b0_data = bottom[0]->gpu_data();
      const Dtype* b1_data = bottom[1]->gpu_data();
      const Dtype* offset_data = bottom[2]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      const int b0_dim = bottom[0]->count(1);
      const int b1_dim = bottom[1]->count(1);
      const int offset_dim = bottom[2]->count(1);
      const int b0_spatial = bottom[0]->count(2);
      const int b1_spatial = bottom[1]->count(2);
      const int top_width = b1_width;
      const int top_spatial = b1_spatial;
      const int top_dim = top_spatial*window_size_*window_size_;
      const int count = top[0]->count();

      DeformCorrelationForward<Dtype>
            // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), channels, channels*sizeof(Dtype)>>>(
          count, b0_data, b1_data, offset_data, num, channels,
          b0_height, b0_width, b1_height, b1_width, displacement_, dilation_, step_h_, step_w_, window_size_,
          b0_dim, b0_spatial, b1_dim, b1_spatial, offset_dim , top_dim, top_spatial, top_width, top_data);
    }
    else
    {
      const int num = bottom[0]->num();
      const int channels = bottom[0]->channels();
      const int b0_height = bottom[0]->height();
      const int b0_width = bottom[0]->width();
      const Dtype* b0_data = bottom[0]->gpu_data();
      const Dtype* offset_data = bottom[1]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      const int b0_dim = bottom[0]->count(1);
      const int offset_dim = bottom[1]->count(1);
      const int b0_spatial = bottom[0]->count(2);
      const int top_width = b0_width;
      const int top_spatial = b0_spatial;
      const int top_dim = top_spatial*window_size_*window_size_;
      const int count = top[0]->count();

      DeformCorrelationForward<Dtype>
            // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), channels, channels*sizeof(Dtype)>>>(
          count, b0_data, b0_data, offset_data, num, channels,
          b0_height, b0_width, b0_height, b0_width, displacement_, dilation_, step_h_, step_w_, window_size_,
          b0_dim, b0_spatial, b0_dim, b0_spatial, offset_dim , top_dim, top_spatial, top_width, top_data);
    }

}

template <typename Dtype>
__global__ void DeformCorrelationBackwardOffset(const int nthreads, 
      const Dtype* top_diff, const Dtype* b0_data, const Dtype* b1_data, const Dtype* offset_data,
      const int num, const int channels, const int b0_height, const int b0_width, 
      const int b1_height, const int b1_width, const int window_size,
      const int displacement, const int dilation, const int step_h, const int step_w, 
      const int b0_dim, const int b0_spatial, const int b1_dim , const int b1_spatial, 
      const int offset_dim, const int top_dim, const int top_spatial, const int top_width, Dtype* offset_diff) {
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const int n = index / top_dim;
    const int out_c = (index % top_dim) / top_spatial;
    const int y = (index % top_spatial) / top_width;
    const int x = index % top_width;
    
    const int b0_x = x*step_w;
    const int b0_y = y*step_h;
    const int dx = (out_c%window_size) - displacement;
    const int dy = out_c/window_size - displacement;

    const Dtype offset_x = offset_data[n*offset_dim+out_c*2*b1_spatial+y*b1_width+x];
    const Dtype offset_y = offset_data[n*offset_dim+(out_c*2+1)*b1_spatial+y*b1_width+x];
    const Dtype h = b0_y + dy*dilation + offset_y;
    const Dtype w = b0_x + dx*dilation + offset_x;

    //bilinear interpolation
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    Dtype hh = h - h_low;
    Dtype lh = 1 - hh;
    Dtype hw = w - w_low;
    Dtype lw = 1 - hw;
    Dtype temp_x = 0.0;
    Dtype temp_y = 0.0;
    if(h_low>=0 && h_high<b0_height && w_low>=0 && w_high < b0_width)
    {
      for(int c=0; c<=channels; c++)
      {
        Dtype data = b1_data[n*b1_dim+c*b1_spatial+y*b1_width+x];

        temp_x -= lh*b0_data[n*b0_dim+c*b0_spatial+h_low*b0_width+w_low]*data;
        temp_y -= lw*b0_data[n*b0_dim+c*b0_spatial+h_low*b0_width+w_low]*data;

        temp_x -= hh*b0_data[n*b0_dim+c*b0_spatial+h_high*b0_width+w_low]*data;
        temp_y += lw*b0_data[n*b0_dim+c*b0_spatial+h_high*b0_width+w_low]*data;

        temp_x += lh*b0_data[n*b0_dim+c*b0_spatial+h_low*b0_width+w_high]*data;
        temp_y -= hw*b0_data[n*b0_dim+c*b0_spatial+h_low*b0_width+w_high]*data;

        temp_x += hh*b0_data[n*b0_dim+c*b0_spatial+h_high*b0_width+w_high]*data;
        temp_y += hw*b0_data[n*b0_dim+c*b0_spatial+h_high*b0_width+w_high]*data;
      }
    }
    
    offset_diff[n*offset_dim+2*out_c*b1_spatial+y*b1_width+x] = temp_x/channels;
    offset_diff[n*offset_dim+(2*out_c+1)*b1_spatial+y*b1_width+x] = temp_y/channels;
  }
}

template <typename Dtype>
__global__ void DeformCorrelationBackward1(const int nthreads, 
      const Dtype* top_diff, const Dtype* b0_data, const Dtype* offset_data, const int num,
      const int channels, const int b0_height, const int b0_width, 
      const int b1_height,const int b1_width, const int window_size,
      const int displacement, const int dilation,
      const int step_h, const int step_w, const int b0_dim, const int b0_spatial,
      const int b1_dim, const int b1_spatial, const int offset_dim, const int top_dim, const int top_spatial,
      const int top_width, Dtype* b1_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / b1_dim;
    const int c = (index % b1_dim) / b1_spatial;
    const int y = (index % b1_spatial) / b1_width;
    const int x = index % b1_width;
    
    Dtype temp = 0.0;

    const int b0_x = x*step_w;
    const int b0_y = y*step_h;
    for(int dx=-1*displacement; dx<=displacement; dx++)
      for(int dy=-1*displacement; dy<=displacement; dy++)
      {
        const int out_c = (dy + displacement)*window_size+(dx + displacement);
        const Dtype offset_x = offset_data[n*offset_dim+out_c*2*b1_spatial+y*b1_width+x];
        const Dtype offset_y = offset_data[n*offset_dim+(out_c*2+1)*b1_spatial+y*b1_width+x];
        const Dtype h = b0_y + dy*dilation + offset_y;
        const Dtype w = b0_x + dx*dilation + offset_x;
        
        //bilinear interpolation
        int h_low = floor(h);
        int w_low = floor(w);
        int h_high = h_low + 1;
        int w_high = w_low + 1;
        Dtype hh = h - h_low;
        Dtype lh = 1 - hh;
        Dtype hw = w - w_low;
        Dtype lw = 1 - hw;
        Dtype data = 0.0;
        if(h_low >=0 && h_high < b0_height && w_low >= 0 && w_high < b0_width)
        {
          Dtype v1 = b0_data[n*b0_dim+c*b0_spatial+h_high*b0_width+w_high];
          Dtype v2 = b0_data[n*b0_dim+c*b0_spatial+h_low*b0_width+w_high];
          Dtype v3 = b0_data[n*b0_dim+c*b0_spatial+h_high*b0_width+w_low];
          Dtype v4 = b0_data[n*b0_dim+c*b0_spatial+h_low*b0_width+w_low];
          data = hh*hw*v1 + lh*hw*v2 + hh*lw*v3 + lw*lh*v4;
        }

        temp += data*top_diff[n*top_dim+out_c*top_spatial+y*top_width+x];

      }
    b1_diff[index] = temp/channels;    
  }
}

template <typename Dtype>
__global__ void ComputeGradientToBlob0(const int nthreads,const Dtype* top_diff, const Dtype* b1_data,
       const Dtype* offset_data, const int channels, const int b0_height, const int b0_width,
       const int b1_height, const int b1_width, const int window_size, const int displacement, const int dilation,
       const int step_h, const int step_w, const int b0_dim, const int b0_spatial, const int b1_dim, const int b1_spatial,
       const int top_spatial, const int top_width, Dtype* container_data)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const int y = index/b1_width;
    const int x = index%b1_width;

    const int b0_x = x*step_w;
    const int b0_y = y*step_h;
    for(int dx=-1*displacement; dx<=displacement; dx++)
      for(int dy=-1*displacement; dy<=displacement; dy++)
      {
        const int out_c = (dy + displacement)*window_size+(dx + displacement);
        const Dtype offset_x = offset_data[out_c*2*b1_spatial+y*b1_width+x];
        const Dtype offset_y = offset_data[(out_c*2+1)*b1_spatial+y*b1_width+x];
        const Dtype h = b0_y + dy*dilation + offset_y;
        const Dtype w = b0_x + dx*dilation + offset_x;
        
        //bilinear interpolation
        int h_low = floor(h);
        int w_low = floor(w);
        int h_high = h_low + 1;
        int w_high = w_low + 1;
        Dtype hh = h - h_low;
        Dtype lh = 1 - hh;
        Dtype hw = w - w_low;
        Dtype lw = 1 - hw;

        Dtype diff = top_diff[out_c*top_spatial+y*top_width+x];
        if(h_low >=0 && h_high < b0_height && w_low >= 0 && w_high < b0_width)
        {
          int locate = h_low*b0_width*b1_height*b1_width+w_low*b1_height*b1_width+y*b1_width+x;
          container_data[locate] = lw*lh*diff/channels;

          locate = h_high*b0_width*b1_height*b1_width+w_low*b1_height*b1_width+y*b1_width+x;
          container_data[locate] = lw*hh*diff/channels;

          locate = h_low*b0_width*b1_height*b1_width+w_high*b1_height*b1_width+y*b1_width+x;
          container_data[locate] = hw*lh*diff/channels;

          locate = h_high*b0_width*b1_height*b1_width+w_high*b1_height*b1_width+y*b1_width+x;
          container_data[locate] = hw*hh*diff/channels;
        }
    }
  }
}

template <typename Dtype>
void DeformCorrelationLayer<Dtype>::Backward_gpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    
    if(!self_)
    {
      const int num = bottom[0]->num();
      const int channels = bottom[0]->channels();
      const int b0_height = bottom[0]->height();
      const int b0_width = bottom[0]->width();
      const int b1_height = bottom[1]->height();
      const int b1_width = bottom[1]->width();
      const Dtype* b0_data = bottom[0]->gpu_data();
      const Dtype* b1_data = bottom[1]->gpu_data();
      const Dtype* offset_data = bottom[2]->gpu_data();
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* b0_diff = bottom[0]->mutable_gpu_diff();
      Dtype* b1_diff = bottom[1]->mutable_gpu_diff();
      Dtype* offset_diff = bottom[2]->mutable_gpu_diff();
      const int b0_dim = bottom[0]->count(1);
      const int b1_dim = bottom[1]->count(1);
      const int offset_dim = bottom[2]->count(1);
      const int b0_spatial = bottom[0]->count(2);
      const int b1_spatial = bottom[1]->count(2);
      const int top_width = b1_width;
      const int top_spatial = b1_spatial;
      const int top_dim = top_spatial*window_size_*window_size_;
      
      for(int i=0; i<=2; i++)
        caffe_gpu_set(bottom[i]->count(), Dtype(0) , bottom[i]->mutable_gpu_diff());

      if (propagate_down[1]) {
        const int b1_count = bottom[1]->count();
        DeformCorrelationBackward1<Dtype>
              // NOLINT_NEXT_LINE(whitespace/operators)
              <<<CAFFE_GET_BLOCKS(b1_count), CAFFE_CUDA_NUM_THREADS>>>(
          b1_count, top_diff, b0_data, offset_data , num, channels,
          b0_height, b0_width, b1_height, b1_width,
          window_size_, displacement_, dilation_, step_h_, step_w_,
          b0_dim, b0_spatial, b1_dim, b1_spatial, offset_dim , top_dim, top_spatial, top_width, b1_diff);
      }
      if (propagate_down[0]) {
        const int b0_count = bottom[0]->count();
        const int b1_count = bottom[1]->count();
        Dtype* container_data = gradient_container_.mutable_gpu_data();

        for(int n=0; n<num; n++)
        {
          caffe_gpu_set(gradient_container_.count(),Dtype(0.0),gradient_container_.mutable_gpu_data());
          ComputeGradientToBlob0<Dtype><<<CAFFE_GET_BLOCKS(b1_spatial),CAFFE_CUDA_NUM_THREADS>>>(
            b1_spatial, top_diff + n*top_dim, b1_data + n*b1_dim, offset_data + n*offset_dim, channels, 
            b0_height, b0_width, b1_height, b1_width, window_size_, displacement_, 
            dilation_, step_h_, step_w_, b0_dim, b0_spatial, b1_dim, b1_spatial, top_spatial, top_width, container_data);

          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels, b0_spatial, b1_spatial,
            1.0, b1_data + n*b1_dim, container_data, 0.0, b0_diff + n*b0_dim);
        }
      }
      if (propagate_down[2]) {
        const int top_count = top[0]->count();
        DeformCorrelationBackwardOffset<Dtype><<<CAFFE_GET_BLOCKS(top_count),CAFFE_CUDA_NUM_THREADS>>>(
          top_count, top_diff, b0_data, b1_data, offset_data, num, channels,
          b0_height, b0_width, b1_height, b1_width,
          window_size_, displacement_, dilation_, step_h_, step_w_,
          b0_dim, b0_spatial, b1_dim, b1_spatial, offset_dim , top_dim, top_spatial, top_width, offset_diff);
      }
    }
    else
    {
      const int num = bottom[0]->num();
      const int channels = bottom[0]->channels();
      const int b0_height = bottom[0]->height();
      const int b0_width = bottom[0]->width();
      const Dtype* b0_data = bottom[0]->gpu_data();
      const Dtype* offset_data = bottom[1]->gpu_data();
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* b0_diff = bottom[0]->mutable_gpu_diff();
      Dtype* offset_diff = bottom[1]->mutable_gpu_diff();
      const int b0_dim = bottom[0]->count(1);
      const int offset_dim = bottom[1]->count(1);
      const int b0_spatial = bottom[0]->count(2);
      const int top_width = b0_width;
      const int top_spatial = b0_spatial;
      const int top_dim = top_spatial*window_size_*window_size_;
      
      for(int i=0; i<=1; i++)
        caffe_gpu_set(bottom[i]->count(), Dtype(0) , bottom[i]->mutable_gpu_diff());

      if (propagate_down[0]) {
        const int b0_count = bottom[0]->count();
        DeformCorrelationBackward1<Dtype>
              // NOLINT_NEXT_LINE(whitespace/operators)
              <<<CAFFE_GET_BLOCKS(b0_count), CAFFE_CUDA_NUM_THREADS>>>(
          b0_count, top_diff, b0_data, offset_data , num, channels,
          b0_height, b0_width, b0_height, b0_width,
          window_size_, displacement_, dilation_, step_h_, step_w_,
          b0_dim, b0_spatial, b0_dim, b0_spatial, offset_dim , top_dim, top_spatial, top_width, b0_diff);

        Dtype* container_data = gradient_container_.mutable_gpu_data();

        for(int n=0; n<num; n++)
        {
          caffe_gpu_set(gradient_container_.count(),Dtype(0.0),gradient_container_.mutable_gpu_data());
          ComputeGradientToBlob0<Dtype><<<CAFFE_GET_BLOCKS(b0_spatial),CAFFE_CUDA_NUM_THREADS>>>(
            b0_spatial, top_diff + n*top_dim, b0_data + n*b0_dim, offset_data + n*offset_dim, channels, 
            b0_height, b0_width, b0_height, b0_width, window_size_, displacement_, 
            dilation_, step_h_, step_w_, b0_dim, b0_spatial, b0_dim, b0_spatial, top_spatial, top_width, container_data);

          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels, b0_spatial, b0_spatial,
            1.0, b0_data + n*b0_dim, container_data, 1.0, b0_diff + n*b0_dim);
        }
      }
      if (propagate_down[1]) {
        const int top_count = top[0]->count();
        DeformCorrelationBackwardOffset<Dtype><<<CAFFE_GET_BLOCKS(top_count),CAFFE_CUDA_NUM_THREADS>>>(
          top_count, top_diff, b0_data, b0_data, offset_data, num, channels,
          b0_height, b0_width, b0_height, b0_width,
          window_size_, displacement_, dilation_, step_h_, step_w_,
          b0_dim, b0_spatial, b0_dim, b0_spatial, offset_dim , top_dim, top_spatial, top_width, offset_diff);
      }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(DeformCorrelationLayer);

}  // namespace caffe