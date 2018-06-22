#include <algorithm>
#include <vector>
#include <cfloat>
#include "caffe/layers/upsample_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void UpsampleForward(const int nthreads,
    const Dtype* b1_data, const int num, const int channels,
    const int b0_height, const int b0_width, const int b1_height, const int b1_width,
    const Dtype w_ratio, const Dtype h_ratio,
    const int b0_dim, const int b0_spatial, const int b1_dim, const int b1_spatial,
    Dtype* const top_data) {

    CUDA_KERNEL_LOOP(index, nthreads)
    {
      const int n = index/b0_dim;
      const int c = (index%b0_dim)/b0_spatial;
      const int y = (index%b0_spatial)/b0_width;
      const int x = index%b0_width;

      const Dtype b1_x = (x+1)/w_ratio - 1.0;
      const Dtype b1_y = (y+1)/h_ratio - 1.0;
      
      const int x_l = int(b1_x) < 0 ? 0 : int(b1_x);
      const int x_h = int(b1_x)+1 > (b1_width-1) ? (b1_width-1): (int(b1_x)+1);
      const Dtype wxl = int(b1_x) < 0 ? 0.0 : (1 - (b1_x - int(b1_x)));
      const Dtype wxh = int(b1_x) + 1 > (b1_width-1) ? 0.0 : ( 1 - wxl);

      const int y_l = int(b1_y) < 0 ? 0 : int(b1_y);
      const int y_h = int(b1_y)+1 > (b1_height-1) ? (b1_height-1): (int(b1_y)+1);
      const Dtype wyl = int(b1_y) < 0 ? 0.0 : (1 - (b1_y - int(b1_y)));
      const Dtype wyh = int(b1_y) + 1 > (b1_height-1) ? 0.0 : ( 1 - wyl);

      Dtype temp = 0.0;
      temp += wxl*wyl*b1_data[n*b1_dim+c*b1_spatial+y_l*b1_width+x_l];
      temp += wxh*wyl*b1_data[n*b1_dim+c*b1_spatial+y_l*b1_width+x_h];
      temp += wxl*wyh*b1_data[n*b1_dim+c*b1_spatial+y_h*b1_width+x_l];
      temp += wxh*wyh*b1_data[n*b1_dim+c*b1_spatial+y_h*b1_width+x_h];
      top_data[index] = temp;
    }
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int b0_height = bottom[0]->height();
    const int b0_width = bottom[0]->width();
    const int b1_height = bottom[1]->height();
    const int b1_width = bottom[1]->width();
    const Dtype* b1_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int b0_dim = bottom[0]->count(1);
    const int b1_dim = bottom[1]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int b1_spatial = bottom[1]->count(2);
    const int count = top[0]->count();

    UpsampleForward<Dtype>
          // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, b1_data, num, channels,
        b0_height, b0_width, b1_height, b1_width,
        w_ratio_, h_ratio_,
        b0_dim, b0_spatial, b1_dim, b1_spatial, top_data);
}


template <typename Dtype>
__global__ void GenerateCoeff(const int nthreads,
  const int b0_height, const int b0_width, const int b1_height, const int b1_width,
  const Dtype w_ratio, const Dtype h_ratio, Dtype* diff){
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const int y = index/b0_width;
    const int x = index%b0_width;

    const Dtype b1_x = (x+1)/w_ratio - 1.0;
    const Dtype b1_y = (y+1)/h_ratio - 1.0;
    
    const int x_l = int(b1_x) < 0 ? 0 : int(b1_x);
    const int x_h = int(b1_x)+1 > (b1_width-1) ? (b1_width-1): (int(b1_x)+1);
    const Dtype wxl = int(b1_x) < 0 ? 0.0 : (1 - (b1_x - int(b1_x)));
    const Dtype wxh = int(b1_x) + 1 > (b1_width-1) ? 0.0 : ( 1 - wxl);

    const int y_l = int(b1_y) < 0 ? 0 : int(b1_y);
    const int y_h = int(b1_y)+1 > (b1_height-1) ? (b1_height-1): (int(b1_y)+1);
    const Dtype wyl = int(b1_y) < 0 ? 0.0 : (1 - (b1_y - int(b1_y)));
    const Dtype wyh = int(b1_y) + 1 > (b1_height-1) ? 0.0 : ( 1 - wyl);

    diff[((y*b0_width+x)*b1_height+y_l)*b1_width+x_l] = wxl*wyl;
    diff[((y*b0_width+x)*b1_height+y_l)*b1_width+x_h] = wxh*wyl;
    diff[((y*b0_width+x)*b1_height+y_h)*b1_width+x_l] = wxl*wyh;
    diff[((y*b0_width+x)*b1_height+y_h)*b1_width+x_h] = wxh*wyh;

  }
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_gpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int b0_height = bottom[0]->height();
    const int b0_width = bottom[0]->width();
    const int b1_height = bottom[1]->height();
    const int b1_width = bottom[1]->width();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* b1_diff = bottom[1]->mutable_gpu_diff();
    Dtype* diff = container_.mutable_gpu_diff();
    const int b0_dim = bottom[0]->count(1);
    const int b1_dim = bottom[1]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int b1_spatial = bottom[1]->count(2);
    
    if (propagate_down[1]) {
      const int count = top[0]->count(2);
      caffe_gpu_set(container_.count(),Dtype(0.0),diff);
      GenerateCoeff<Dtype>
            // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count,  b0_height, b0_width, b1_height, b1_width,
        w_ratio_, h_ratio_, diff);

      for(int n=0; n<num; n++)
      {
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, channels, b1_spatial, b0_spatial, Dtype(1.0),
          top_diff + n*b0_dim, diff, Dtype(0.0), b1_diff + n*b1_dim);
      }
    }
    if (propagate_down[0]) {
      NOT_IMPLEMENTED;
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(UpsampleLayer);

}  // namespace caffe