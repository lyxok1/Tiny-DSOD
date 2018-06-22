#include <vector>

#include "caffe/layers/enhance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EnhanceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CorrelationParameter corr_param = this->layer_param_.correlation_param();
  CHECK(corr_param.has_displacement()) << "Displacement must be specified for correlation";
  kernel_size_ = corr_param.kernel_size();
  displacement_ = corr_param.displacement();
  if(corr_param.has_dilation())
    dilation_ = corr_param.dilation();
  else
    dilation_ = 1;
  CHECK_EQ(kernel_size_, 1) << "Current version only support kernel_size of 1";

  // setup correlation layer
  correlation_.reset(new CorrelationLayer<Dtype>(this->layer_param_));
  corr_top_vec_.clear();
  corr_top_vec_.push_back(&corr_);
  correlation_->SetUp(bottom, corr_top_vec_);

  // setup softmax layer
  softmax_.reset(new SoftmaxLayer<Dtype>(this->layer_param_));
  soft_top_vec_.clear();
  soft_top_vec_.push_back(&soft_);
  softmax_->SetUp(corr_top_vec_, soft_top_vec_);
}

template <typename Dtype>
void EnhanceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   top[0]->ReshapeLike(*bottom[0]);
   correlation_->Reshape(bottom, corr_top_vec_);
   softmax_->Reshape(corr_top_vec_, soft_top_vec_);
}

template <typename Dtype>
void EnhanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void EnhanceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(EnhanceLayer);
#endif

INSTANTIATE_CLASS(EnhanceLayer);
REGISTER_LAYER_CLASS(Enhance);

}  // namespace caffe
