#include <vector>
#include <algorithm>
#include <cfloat>

#include "caffe/layers/deform_sample_layer.hpp"
#include "caffe/util/math_functions.hpp"

//define floor int conversion to avoid negative numbers
#define TO_FLOOR_INT(x) x > 0.0 ? int(x) : (int(x) - 1)

namespace caffe {

template <typename Dtype>
__global__ void GetBilinearCoeffKernel(const int nthread, const Dtype* offset,
  const int num, const int channels, const int height, const int width, const int num_samples_per_loc, 
  const int spatial, Dtype* coeff)
{
  CUDA_KERNEL_LOOP(index, nthread)
  {
    const int p = index % spatial;
    const int s = (index / spatial) % (spatial);
    const int g = (index / spatial / spatial) % num_samples_per_loc;
    const int n = index / spatial / spatial / num_samples_per_loc;
    const int px = s % width;
    const int py = s / width;

    const int x = p % width;
    const int y = p / width;
    
    const Dtype dx = offset[((n*2*num_samples_per_loc+g*2)*height+y)*width+x];
    const Dtype dy = offset[((n*2*num_samples_per_loc+g*2+1)*height+y)*width+x];
    const Dtype offset_x = x+dx;
    const Dtype offset_y = y+dy;

    coeff[index] = (max(0.0, 1-abs(offset_x - px)))*(max(0.0,1-abs(offset_y-py)));
  }
}

template <typename Dtype>
void DeformSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
      const int num = bottom[0]->num();
      const int channels = bottom[0]->channels();
      const int height = bottom[0]->height();
      const int width = bottom[0]->width();

      const Dtype* feature = bottom[0]->gpu_data();
      const Dtype* offset = bottom[1]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();

      const int count = bi_coeff_.count();
      Dtype* bi_coeff = bi_coeff_.mutable_gpu_data();
      const int spatial = height*width;

      caffe_gpu_set(count, Dtype(0.0), bi_coeff);

      GetBilinearCoeffKernel<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>
        (count, offset, num, channels, height, width, num_samples_per_loc_, spatial, bi_coeff);

      
      const int b0_dim = bottom[0]->count(1);
      const int coeff_dim = bi_coeff_.count(2);
      const int top_dim = top[0]->count(1);
      const int groups = channels / num_samples_per_loc_;
      const int group_dim = b0_dim / num_samples_per_loc_;

      for(int n=0; n < num; n++)
      {
        for(int g=0; g < num_samples_per_loc_; g++)
        {
          const int g_idx = n*num_samples_per_loc_ + g;
          caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, groups, spatial,
                spatial, Dtype(1.0), feature + n*b0_dim + g*group_dim, bi_coeff + g_idx*coeff_dim,
                Dtype(0), top_data + n*top_dim + g*group_dim);
        }
      }
}

template <typename Dtype>
__global__ void BackwardGradientKernel(const int nthread, const Dtype* coeff_diff,
  const Dtype* offset, const int height, const int width, const int spatial, 
  const int num_samples_per_loc, Dtype* diff)
{
  CUDA_KERNEL_LOOP(index, nthread)
  {
    const int n = index / height / width / num_samples_per_loc;
    const int x = index % width;
    const int y = (index / width) % height;
    const int g = (index / height / width) % num_samples_per_loc;

    const int s = index % spatial;

    const Dtype dx = offset[((n*2*num_samples_per_loc+g*2)*height+y)*width+x];
    const Dtype dy = offset[((n*2*num_samples_per_loc+g*2+1)*height+y)*width+x];
    const Dtype offset_x = x+dx;
    const Dtype offset_y = y+dy;

    const int xl = TO_FLOOR_INT(offset_x);
    const int xh = xl + 1;
    const int yl = TO_FLOOR_INT(offset_y);
    const int yh = yl + 1;

    Dtype temp_x = 0.0;
    Dtype temp_y = 0.0;
    if(xl<=(width-1) && xl>=0 && yl<=(height-1) && yl>=0)
    {
      temp_x -= (yh-offset_y)*coeff_diff[((n*num_samples_per_loc+g)*spatial+yl*width+xl)*spatial+s];
      temp_y -= (xh-offset_x)*coeff_diff[((n*num_samples_per_loc+g)*spatial+yl*width+xl)*spatial+s];
    }

    if(xh<=(width-1) && xh>=0 && yl<=(height-1) && yl>=0)
    {
      temp_x += (yh-offset_y)*coeff_diff[((n*num_samples_per_loc+g)*spatial+yl*width+xh)*spatial+s];
      temp_y -= (offset_x-xl)*coeff_diff[((n*num_samples_per_loc+g)*spatial+yl*width+xh)*spatial+s];
    }

    if(xl<=(width-1) && xl>=0 && yh<=(height-1) && yh>=0)
    {
      temp_x -= (offset_y-yl)*coeff_diff[((n*num_samples_per_loc+g)*spatial+yh*width+xl)*spatial+s];
      temp_y += (xh-offset_x)*coeff_diff[((n*num_samples_per_loc+g)*spatial+yh*width+xl)*spatial+s];
    }

    if(xh<=(width-1) && xh>=0 && yh<=(height-1) && yh>=0)
    {
      temp_x += (offset_y-yl)*coeff_diff[((n*num_samples_per_loc+g)*spatial+yh*width+xh)*spatial+s];
      temp_y += (offset_x-xl)*coeff_diff[((n*num_samples_per_loc+g)*spatial+yh*width+xh)*spatial+s];
    }
    
    diff[((n*2*num_samples_per_loc+g*2)*height+y)*width+x] = temp_x;
    diff[((n*2*num_samples_per_loc+g*2+1)*height+y)*width+x] = temp_y;
  }
}

template <typename Dtype>
void DeformSampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      const int num = bottom[0]->num();
      const int channels = bottom[0]->channels();
      const int height = bottom[0]->height();
      const int width = bottom[0]->width();

      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* feature = bottom[0]->gpu_data();
      const Dtype* offset = bottom[1]->gpu_data();
      Dtype* feature_diff = bottom[0]->mutable_gpu_diff();
      Dtype* offset_diff = bottom[1]->mutable_gpu_diff();

      const Dtype* bi_coeff = bi_coeff_.gpu_data();
      Dtype* bi_coeff_diff = bi_coeff_.mutable_gpu_diff();

      const int spatial = height*width;
      const int b0_dim = bottom[0]->count(1);
      const int coeff_dim = bi_coeff_.count(2);
      const int top_dim = top[0]->count(1);

      const int groups = channels / num_samples_per_loc_;
      const int group_dim = b0_dim / num_samples_per_loc_;

      if(propagate_down[0])
      {
        //backward diff from top to bottom features
        for(int n=0; n<num; n++)
        {
          for(int g=0; g<num_samples_per_loc_; g++)
          {
            const int g_idx = n*num_samples_per_loc_ + g;
            caffe_gpu_gemm(CblasNoTrans, CblasTrans, groups, spatial,
              spatial, Dtype(1.0), top_diff + n*top_dim + g*group_dim, bi_coeff + g_idx*coeff_dim,
              Dtype(0.0), feature_diff + n*b0_dim + g*group_dim);
          }
        }
      }

      if(propagate_down[1])
      {
        //backward diff from top to bilinear coefficient
        for(int n=0; n<num; n++)
          for(int g=0; g<num_samples_per_loc_; g++)
          {
            const int g_idx = n*num_samples_per_loc_ + g;
            caffe_gpu_gemm(CblasTrans, CblasNoTrans, spatial,
                spatial, groups, Dtype(1.0), feature + n*b0_dim + g*group_dim, top_diff + n*top_dim + g*group_dim, 
                Dtype(0.0), bi_coeff_diff + g_idx*coeff_dim);
          }

        //backward diff from bilinear coefficient to offsets
        const int count = bottom[1]->count()/2;
         BackwardGradientKernel<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>
            (count, bi_coeff_diff ,offset, height, width, spatial, 
              num_samples_per_loc_, offset_diff);
      }
}

INSTANTIATE_LAYER_GPU_FUNCS(DeformSampleLayer);

}  // namespace caffe
