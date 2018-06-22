#include <algorithm>
#include <vector>
#include <cmath>
#include "caffe/filler.hpp"
#include "caffe/layers/deform_correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DeformCorrelationLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CorrelationParameter corr_param = this->layer_param_.correlation_param();
  
  CHECK(corr_param.has_displacement()) << "Displacement must be specified for correlation";
  CHECK_GE(bottom.size(), 2) << "DeformCorrelation layer only receive at least two blobs as input (data and offset)";
  CHECK_LE(bottom.size(), 3) << "DeformCorrelation layer only receive at most three blobs as input";
  int offset_idx;
  if(bottom.size()==3)
  {
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    self_ = false;
    offset_idx = 2;
  }
  else
  {
    self_ = true;
    offset_idx = 1;
  }

  displacement_ = corr_param.displacement();
  window_size_ = displacement_*2+1;

  if(corr_param.has_dilation())
    dilation_ = 1;
  else
    dilation_ = corr_param.dilation();

  int offset_nums = bottom[offset_idx]->channels();
  CHECK_EQ(offset_nums, window_size_*window_size_*2) << "Offsets shape mismatch";

  if(!self_)
  {
    const int b0_w = bottom[0]->width();
    const int b1_w = bottom[1]->width();
    const int b0_h = bottom[0]->height();
    const int b1_h = bottom[1]->height();

    CHECK_GE(b0_w, b1_w);
    CHECK_GE(b0_h, b1_h);

    if(b0_w==b1_w)
      step_w_ = 1;
    else
      step_w_ = b0_w/b1_w;

    if(b0_h==b1_h)
      step_h_ = 1;
    else
      step_h_ = b0_h/b1_h;
  }
  else
  {
    step_w_ = 1;
    step_h_ = 1;
  }
}

template <typename Dtype>
void DeformCorrelationLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  int idx = 0;
  if(!self_)
    idx = 1;
  top_shape.push_back(bottom[idx]->num());
  top_shape.push_back(window_size_*window_size_);
  top_shape.push_back(bottom[idx]->height());
  top_shape.push_back(bottom[idx]->width());
  top[0]->Reshape(top_shape);

  if(!self_)
  {
    vector<int> container_shape;
    container_shape.push_back(bottom[0]->height());
    container_shape.push_back(bottom[0]->width());
    container_shape.push_back(bottom[1]->height());
    container_shape.push_back(bottom[1]->width());
    gradient_container_.Reshape(container_shape);
  }
  else
  {
    vector<int> container_shape;
    container_shape.push_back(bottom[0]->height());
    container_shape.push_back(bottom[0]->width());
    container_shape.push_back(bottom[0]->height());
    container_shape.push_back(bottom[0]->width());
    gradient_container_.Reshape(container_shape);
  }
}

template <typename Dtype>
void DeformCorrelationLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void DeformCorrelationLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(DeformCorrelationLayer);
#endif

INSTANTIATE_CLASS(DeformCorrelationLayer);
REGISTER_LAYER_CLASS(DeformCorrelation);

}  // namespace caffe
