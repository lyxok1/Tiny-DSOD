#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CrossEntropyForward(const int nthread, const Dtype* prob, const Dtype* labels,
 Dtype* entropy, Dtype alpha)
{
  CUDA_KERNEL_LOOP(index, nthread)
  {
    const Dtype label = labels[index];
    const Dtype p = max(prob[index], Dtype(kLOG_THRESHOLD));
    entropy[index] = label*log(p)*(1-alpha)+(1-label)*log(1-p)*alpha;
  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  Dtype* entropy = bottom[1]->mutable_gpu_diff();
  
  const int count = bottom[0]->count();

  Dtype loss = 0;
  Dtype alpha = 0.0;

  if(compensate_imbalance_)
  {
    caffe_gpu_asum(count, bottom_label, &alpha); // the positive ratio
    alpha /= count;
  }
  else
    alpha = 0.5;

  CrossEntropyForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (count, bottom_data, bottom_label, entropy, alpha);

  caffe_gpu_asum(count, entropy, &loss);

  const int num = top[0]->num();
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
__global__ void CrossEntropyBackward(const int nthread, const Dtype* prob, const Dtype* labels,
 Dtype* diff, Dtype alpha, Dtype scale)
{
  CUDA_KERNEL_LOOP(index, nthread)
  {
    const Dtype label = labels[index];
    const Dtype p = max(prob[index], Dtype(kLOG_THRESHOLD));
    diff[index] = scale * ((1-alpha)*label/p - alpha*(1-label)/(1-p));
  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_label = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    const int count = bottom[0]->count();
    caffe_gpu_set(count, Dtype(0), bottom_diff);

    const int num = top[0]->num();
    const Dtype scale =  -1 * top[0]->cpu_diff()[0] / num;
    Dtype alpha = 0.0;

    if(compensate_imbalance_)
    {
      caffe_gpu_asum(count, bottom_label, &alpha); // the positive ratio
      alpha /= count;
    }
    else
      alpha = 0.5;

    CrossEntropyBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, bottom_data, bottom_label, bottom_diff, alpha, scale);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CrossEntropyLossLayer);

}  // namespace caffe
