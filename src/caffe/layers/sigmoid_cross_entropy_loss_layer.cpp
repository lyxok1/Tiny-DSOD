#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (this->layer_param_.loss_param().has_normalization()) {
    normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }
  if(this->layer_param_.loss_param().has_focal_loss_param())
  {
    focal_loss_ = this->layer_param_.loss_param().focal_loss();
    compensate_imbalance_ = this->layer_param_.loss_param().focal_loss_param().compensate_imbalance();
    gamma_ = this->layer_param_.loss_param().focal_loss_param().gamma();
  }
  else
  {
    focal_loss_ = false;
    compensate_imbalance_ = false;
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

// TODO(shelhamer) loss normalization should be pulled up into LossLayer,
// instead of duplicated here and in SoftMaxWithLossLayer
template <typename Dtype>
Dtype SigmoidCrossEntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  int valid_count = 0;
  Dtype loss = 0;
  Dtype alpha_ = 0.0;
  if(compensate_imbalance_)
  {
    alpha_ = caffe_cpu_asum(bottom[1]->count(),target)/Dtype((bottom[1]->count()));
  }

  if(!focal_loss_&&!compensate_imbalance_)
    for (int i = 0; i < bottom[0]->count(); ++i) {
      const int target_value = static_cast<int>(target[i]);
      if (has_ignore_label_ && target_value == ignore_label_) {
        continue;
      }
      loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
      ++valid_count;
    }

  if(!focal_loss_&&compensate_imbalance_)
    for (int i = 0; i < bottom[0]->count(); ++i) {
      const int target_value = static_cast<int>(target[i]);
      if (has_ignore_label_ && target_value == ignore_label_) {
        continue;
      }
      loss -= (input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)))) * 
          (target[i]==1?(1/alpha_):(1/(1-alpha_)));
      ++valid_count;
    }

  if(focal_loss_&&!compensate_imbalance_)
    for (int i = 0; i < bottom[0]->count(); ++i) {
      const int target_value = static_cast<int>(target[i]);
      if (has_ignore_label_ && target_value == ignore_label_) {
        continue;
      }
      loss -= (input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)))) / 
          pow(1+exp((2*target[i]-1)*input_data[i]),gamma_);
      ++valid_count;
    }

  if(focal_loss_&&compensate_imbalance_)
    for (int i = 0; i < bottom[0]->count(); ++i) {
      const int target_value = static_cast<int>(target[i]);
      if (has_ignore_label_ && target_value == ignore_label_) {
        continue;
      }
      loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))) * 
          (target[i]==1?(1/alpha_):(1/(1-alpha_))) / 
          pow(1+exp((2*target[i]-1)*input_data[i]),gamma_);
      ++valid_count;
    }

  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    if(!compensate_imbalance_&&!focal_loss_)
    {
      caffe_sub(count, sigmoid_output_data, target, bottom_diff);
      if (has_ignore_label_) {
        for (int i = 0; i < count; ++i) {
          const int target_value = static_cast<int>(target[i]);
          if (target_value == ignore_label_) {
            bottom_diff[i] = 0;
          }
        }
      }
    }

    if(compensate_imbalance_&&!focal_loss_)
    {
      // Zero out gradient of ignored targets.
      if (has_ignore_label_) {
        for (int i = 0; i < count; ++i) {
          const int target_value = static_cast<int>(target[i]);
          if (target_value == ignore_label_) 
            bottom_diff[i] = 0;
          else
            bottom_diff[i] = (sigmoid_output_data[i] - target[i])/((2*alpha_-1)*target[i]+1-alpha_);
        }
      }
    }

    if(!compensate_imbalance_&&focal_loss_)
    {
      // Zero out gradient of ignored targets.
      if (has_ignore_label_) {
        for (int i = 0; i < count; ++i) {
          const int target_value = static_cast<int>(target[i]);
          if (target_value == ignore_label_) {
            bottom_diff[i] = 0;
          }
          else if(target[i]==0)
            bottom_diff[i] = pow(sigmoid_output_data[i],gamma_)*(sigmoid_output_data[i]-gamma_*log(1-sigmoid_output_data[i])*(1-sigmoid_output_data[i]));
          else
            bottom_diff[i] = pow(1-sigmoid_output_data[i],gamma_)*
                            (gamma_*log(sigmoid_output_data[i])*sigmoid_output_data[i]-sigmoid_output_data[i]-1+sigmoid_output_data[i]);
        }
      }
    }

    if(compensate_imbalance_&&focal_loss_)
    {
      // Zero out gradient of ignored targets.
      if (has_ignore_label_) {
        for (int i = 0; i < count; ++i) {
          const int target_value = static_cast<int>(target[i]);
          if (target_value == ignore_label_) {
            bottom_diff[i] = 0;
          }
          else if(target[i]==0)
            bottom_diff[i] = pow(sigmoid_output_data[i],gamma_)*(sigmoid_output_data[i]-gamma_*log(1-sigmoid_output_data[i])*(1-sigmoid_output_data[i]));
          else
            bottom_diff[i] = pow(1-sigmoid_output_data[i],gamma_)*
                            (gamma_*log(sigmoid_output_data[i])*sigmoid_output_data[i]-sigmoid_output_data[i]-1+sigmoid_output_data[i]);

          bottom_diff[i] = bottom_diff[i]/((2*alpha_-1)*target[i]+1-alpha_);
        }
      }
    }

    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
