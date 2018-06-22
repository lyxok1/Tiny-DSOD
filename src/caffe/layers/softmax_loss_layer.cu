#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts, bool focal_loss, bool compensate_imbalance,
          Dtype gamma, const int background_label_id,const Dtype alpha, const bool is_condition) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      if(!compensate_imbalance)
        if(!focal_loss)
            loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                          Dtype(FLT_MIN)));
        else
            loss[index] = -1*pow(1 - prob_data[n * dim + label_value * spatial_dim + s],gamma)*log(max(prob_data[n * dim + label_value * spatial_dim + s],
                               Dtype(FLT_MIN)));
      else
        if(!focal_loss)
          if(label_value==background_label_id)
          	loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                          	Dtype(FLT_MIN)))*(1-alpha);
          else
 		loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
 				Dtype(FLT_MIN)))*alpha;
        else
	  if(label_value==background_label_id)
          	loss[index] =-1*pow(1 - prob_data[n * dim + label_value * spatial_dim + s],gamma)*log(max(prob_data[n * dim + label_value * spatial_dim + s],
                               Dtype(FLT_MIN)))*(1-alpha);
          else
		loss[index] =-1*pow(1 - prob_data[n * dim + label_value * spatial_dim + s],gamma)*log(max(prob_data[n * dim + label_value * spatial_dim + s],
                               Dtype(FLT_MIN)))*alpha;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void FindLabels(const int nthreads, const Dtype* labels, const int label_value, Dtype* output, bool has_ignore_label_, const int ignore_label_)
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
    const int label_val = static_cast<int>(labels[index]);
    if (!(has_ignore_label_ && label_value == ignore_label_))
    {
      if(label_val==label_value)
        output[index] = 1.0;
      else
        output[index] = 0.0;
    }
    else
      output[index] = 0.0;
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, these memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();

  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts, 
      focal_loss_,compensate_imbalance_,gamma_,background_label_id_,alpha_, is_condition_);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
      normalization_, outer_num_, inner_num_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const Dtype* prob_data, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_, 
          const int ignore_label_, Dtype* counts,bool focal_loss,bool compensate_imbalance,
          Dtype gamma, const int background_label_id, const Dtype alpha, const bool is_condition) {

  //const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int c = (index % dim) / spatial_dim;
    const int s = (index % dim) % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      bottom_diff[n * dim + c * spatial_dim + s] = 0;
      counts[index] = 0;
    } else {
      if(!compensate_imbalance)
      {
        if(!focal_loss)
        {
          if(!is_condition)
          {
            if(c==label_value)
              bottom_diff[n * dim + c * spatial_dim + s] = prob_data[n * dim + label_value * spatial_dim + s] - 1;
            else
              bottom_diff[n * dim + c * spatial_dim + s] = prob_data[n * dim + c * spatial_dim + s];
          }
          else
          {
            if(c==0)
              if(c==label_value)
                bottom_diff[n * dim + c * spatial_dim + s] = prob_data[n * dim + c * spatial_dim + s] - 1;
              else
                bottom_diff[n * dim + c * spatial_dim + s] = prob_data[n * dim + c * spatial_dim + s];
            else
            {
              if(c==label_value&&label_value!=0)
                bottom_diff[n * dim + c * spatial_dim + s] = prob_data[n * dim + c * spatial_dim + s]/(1-prob_data[n*dim+s]) - 1;
              else if(c!=label_value&&label_value!=0)
                bottom_diff[n * dim + c * spatial_dim + s] = prob_data[n * dim + c * spatial_dim + s]/(1-prob_data[n*dim+s]);
              else
                bottom_diff[n * dim + c * spatial_dim + s] = 0.0;
            }
          }
        }
        else
        {
          Dtype base_p = prob_data[n*dim+label_value*spatial_dim+s];
          Dtype base = pow(max(1-base_p,Dtype(FLT_MIN)),gamma-1)*(base_p+gamma*base_p*log(base_p)-1);
          if(c==label_value)
              bottom_diff[n * dim + c * spatial_dim + s] = base*(1-base_p);
          else
              bottom_diff[n * dim + c * spatial_dim + s] = -1*base*(prob_data[n*dim+c*spatial_dim+s]);
        }
      }
      else
      {
        if(!focal_loss)
        {
          if(c==label_value)
            bottom_diff[n * dim + c * spatial_dim + s] = (prob_data[n * dim + label_value * spatial_dim + s] - 1)*(label_value==background_label_id?(1-alpha):alpha);
          else
            bottom_diff[n * dim + c * spatial_dim + s] = prob_data[n * dim + c * spatial_dim + s]*(label_value==background_label_id?(1-alpha):alpha);
        }
        else
        {
          Dtype base_p = prob_data[n*dim+label_value*spatial_dim+s];
          Dtype base = pow(max(1-base_p,Dtype(FLT_MIN)),gamma-1)*(base_p+gamma*base_p*log(base_p)-1);
          if(c==label_value)
              bottom_diff[n * dim + c * spatial_dim + s] = base*(1-base_p)*(label_value==background_label_id?(1-alpha):alpha);
          else
              bottom_diff[n * dim + c * spatial_dim + s] = -1*base*prob_data[n*dim+c*spatial_dim+s]*(label_value==background_label_id?(1-alpha):alpha);
        }
      }
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void ValidCountGPU(const int nthreads,
          const Dtype* label, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_,Dtype* counts) {
  CUDA_KERNEL_LOOP(index,nthreads)
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      counts[index] = 0;
    } else {
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    //caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    int nthreads = prob_.count();
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff, prob_data,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts,
        focal_loss_,compensate_imbalance_,gamma_,background_label_id_,alpha_, is_condition_);

    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, outer_num_, inner_num_, valid_count);
    const Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
