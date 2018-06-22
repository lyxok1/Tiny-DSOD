#include <cfloat>
#include <vector>

#include "caffe/layers/random_shuffle_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RandomShuffleForward(const int n, const Dtype* in,
  const Dtype* rn_data, const int dim, const int spatial, const int width,
  const int axis, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int num = index/dim;
    const int channel = (index%dim)/spatial;
    int shuffle_index = 0;
    int h = 0;
    int w = 0;
    int s = 0;
    switch(axis)
    {
      case 1:
        s = index%spatial;
        shuffle_index = static_cast<int>(rn_data[channel]);
        out[num*dim+shuffle_index*spatial+s] = in[index];
        break;
      case 2:
        h = (index%spatial)/width;
        w = index%width;
        shuffle_index = static_cast<int>(rn_data[h]);
        out[num*dim+channel*spatial+shuffle_index*width+w] = in[index];
        break;
      case 3:
        h = (index%spatial)/width;
        w = index%width;
        shuffle_index = static_cast<int>(rn_data[w]);
        out[num*dim+channel*spatial+shuffle_index*width+shuffle_index] = in[index];
        break;
    }
  }
}

template <typename Dtype>
__global__ void RandomShuffleBackward(const int n, Dtype* in,
  const Dtype* rn_data, const int dim, const int spatial, const int width,
  const int axis, const Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int num = index/dim;
    const int channel = (index%dim)/spatial;
    int shuffle_index = 0;
    int h = 0;
    int w = 0;
    int s = 0;
    switch(axis)
    {
      case 1:
        s = index%spatial;
        shuffle_index = static_cast<int>(rn_data[channel]);
        in[index] = out[num*dim+shuffle_index*spatial+s];
        break;
      case 2:
        h = (index%spatial)/width;
        w = index%width;
        shuffle_index = static_cast<int>(rn_data[h]);
        in[index] = out[num*dim+channel*spatial+shuffle_index*width+w];
        break;
      case 3:
        h = (index%spatial)/width;
        w = index%width;
        shuffle_index = static_cast<int>(rn_data[w]);
        in[index] = out[num*dim+channel*spatial+shuffle_index*width+shuffle_index];
        break;
    }
  }
}

template <typename Dtype>
void RandomShuffleLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for(int i=0;i<top.size();++i)
  {
    const int count = top[i]->count();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const Dtype* rn_data = this->blobs_[0]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    const int spatial = bottom[i]->count(2);
    const int dim = bottom[i]->count(1);
    const int width = bottom[i]->width();

    RandomShuffleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, rn_data, dim, spatial, width, axis_, top_data);
  }
}

template <typename Dtype>
void RandomShuffleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for(int i=0;i<top.size();++i)
  {
    if(propagate_down[0])
    {
    	const int count = top[i]->count();
    	Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    	const Dtype* rn_data = this->blobs_[0]->gpu_data();
    	const Dtype* top_diff = top[i]->gpu_data();
	
    	const int spatial = bottom[i]->count(2);
    	const int dim = bottom[i]->count(1);
    	const int width = bottom[i]->width();
    
    RandomShuffleBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_diff, rn_data, dim, spatial, width, axis_, top_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RandomShuffleLayer);

}  // namespace caffe
