#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
  if(this->layer_param_.loss_param().has_focal_loss_param())
    compensate_imbalance_ = this->layer_param_.loss_param().focal_loss_param().compensate_imbalance();
  else
    compensate_imbalance_ = false;
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), bottom[0]->channels());
  CHECK_EQ(bottom[1]->height(), bottom[0]->height());
  CHECK_EQ(bottom[1]->width(), bottom[0]->width());
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  
  Dtype loss = 0;

  Dtype alpha = 0.0;

  if(compensate_imbalance_)
    alpha = caffe_cpu_asum(bottom[1]->count(), bottom_label)/Dtype((bottom[1]->count())); // the positive ratio
  else
    alpha = 0.5;

  for (int i = 0; i < bottom[0]->count(); ++i) {
    int label = static_cast<int>(bottom_label[i]);
    Dtype prob = std::max(
        bottom_data[i], Dtype(kLOG_THRESHOLD));

    loss -= label*log(prob)*(1-alpha)+(1-label)*log(1-prob)*(alpha);

  }
  const int num = top[0]->num();
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    Dtype alpha = 0.0;

    if(compensate_imbalance_)
      alpha = caffe_cpu_asum(bottom[1]->count(), bottom_label)/Dtype((bottom[1]->count())); // the positive ratio
    else
      alpha = 0.5;

    const int num = top[0]->num();
    const Dtype scale = - top[0]->cpu_diff()[0] / num;
    for (int i = 0; i < bottom[0]->count(); ++i) {
      int label = static_cast<int>(bottom_label[i]);
      Dtype prob = std::max(
          bottom_data[i], Dtype(kLOG_THRESHOLD));
      bottom_diff[i] = scale * ((1-alpha)*label/prob - alpha*(1-label)/(1-prob));
    }
  }
}

INSTANTIATE_CLASS(CrossEntropyLossLayer);
REGISTER_LAYER_CLASS(CrossEntropyLoss);

}  // namespace caffe
