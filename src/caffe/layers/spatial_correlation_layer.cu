#include <vector>
#include <algorithm>
#include <cfloat>
#include "caffe/layers/spatial_correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void CorrelationForward(const int nthreads,
    const Dtype* bottom_data, const int num, const int channels,
    const int bottom_height, const int bottom_width, const int top_height, const int top_width,
    Dtype* const top_data) {

    CUDA_KERNEL_LOOP(index, nthreads)
    {
      const int n = index / top_width / top_height /channels;
      const int c = (index / top_width / top_height) % channels;
      const int y = (index / top_width)%top_height;
      const int x = index % top_width;

      if(x > (2*bottom_width-2) || y > (2*bottom_height-2))
      {
        top_data[index] = 0.0; // zero padded for the outermost elements
      }
      else
      {
        const int wstart0 = x < bottom_width ? ( bottom_width - x - 1 ) : 0;
        const int wend0 = x < bottom_width ? ( bottom_width - 1 ) : (2*bottom_width - 2 - x);
        const int hstart0 = y < bottom_height ? ( bottom_height - y - 1 ) : 0;
        const int hend0 = y < bottom_height ? ( bottom_height - 1 ) : (2*bottom_height - 2 - y);

        const int wstart1 = x < bottom_width ? 0 : ( x - bottom_width + 1);
        const int hstart1 = y < bottom_height ? 0 : ( y - bottom_height + 1);      

        Dtype temp = 0.0;
        const int base = n*channels + c;
        const Dtype normalizer = (wend0 - wstart0+1)*(hend0 - hstart0+1);
        for(int i=wstart0; i <= wend0; i++)
          for(int j=hstart0; j<= hend0; j++)
          {
            temp += bottom_data[(base*bottom_height+j)*bottom_width+i]
                    *bottom_data[(base*bottom_height+hstart1+j-hstart0)*bottom_width+i-wstart0+wstart1];
          }
        top_data[index] = temp/normalizer;
      }
    }
}

template <typename Dtype>
void SpatialCorrelationLayer<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int bottom_height = bottom[0]->height();
    const int bottom_width = bottom[0]->width();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int top_width = (2*bottom_width);
    const int top_height = (2*bottom_width);
    const int count = top[0]->count();

    CorrelationForward<Dtype>
          // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, num, channels, bottom_height, bottom_width, top_height, top_width,
        top_data);
}

template <typename Dtype>
__global__ void CorrelationBackward(const int nthreads, 
      const Dtype* bottom_data, const Dtype* top_diff,
      const int channels, const int bottom_height, const int bottom_width,
      const int top_height, const int top_width, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const int n = index / bottom_width / bottom_height /channels;
    const int c = (index / bottom_width / bottom_height) % channels;
    const int y = (index / bottom_width) % bottom_height;
    const int x = index % bottom_width;
    
    Dtype temp = 0.0;

    const int base = n*channels + c;

    for(int i=0; i<bottom_width; i++)
      for(int j=0; j<bottom_height; j++)
      {
          const int dx = bottom_width - 1 - i + x;
          const int dy = bottom_height - 1 - j + y;

          const int wstart0 = dx < bottom_width ? ( bottom_width - dx - 1 ) : 0;
          const int wend0 = dx < bottom_width ? ( bottom_width - 1 ) : (2*bottom_width - 2 - dx);
          const int hstart0 = dy < bottom_height ? ( bottom_height - dy - 1 ) : 0;
          const int hend0 = dy < bottom_height ? ( bottom_height - 1 ) : (2*bottom_height - 2 - dy);
          const Dtype normalizer = (wend0 - wstart0+1)*(hend0 - hstart0+1);

          temp += (top_diff[(base*top_height+dy)*top_width+dx]
                  +top_diff[(base*top_height+2*bottom_height-dy-2)*top_width+2*bottom_width-dx-2])
                  *bottom_data[(base*bottom_height+j)*bottom_width+i]/normalizer;
          // there are two chances that two specific elements are multiplied and the normalizers are the same
      }
      
    bottom_diff[index] = temp;
  }
}

template <typename Dtype>
void SpatialCorrelationLayer<Dtype>::Backward_gpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    if(propagate_down[0])
    {
      const int num = bottom[0]->num();
      const int channels = bottom[0]->channels();
      const int bottom_height = bottom[0]->height();
      const int bottom_width = bottom[0]->width();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const int top_width = 2*bottom_width;
      const int top_height = 2*bottom_width;
      const int count = bottom[0]->count();

      CorrelationBackward<Dtype>
            // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, top_diff, channels, bottom_height, bottom_width, top_height, top_width,
          bottom_diff);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialCorrelationLayer);

}  // namespace caffe