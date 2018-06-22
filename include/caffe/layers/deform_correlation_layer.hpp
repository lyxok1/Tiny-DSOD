#ifndef CAFFE_DEFORM_CORRELATION_LAYER_HPP_
#define CAFFE_DEFORM_CORRELATION_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
class DeformCorrelationLayer : public Layer<Dtype> {
 public:
  explicit DeformCorrelationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "DeformCorrelation"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  int displacement_;
  int window_size_;
  int step_w_;
  int step_h_;
  int dilation_;

  Blob<Dtype> gradient_container_; 
  // Since it is hard to compute gradient from top to bottom[0] directly in a single kernel
  // we use an auxiliary blob to help computation
  
  //TODO: combine kernel_size with deformable-correlation
  //int kernel_size_;
  bool self_;//bool value to indicate if self-correlation (true) or cross-correlation 
};

}  // namespace caffe


#endif  // CAFFE_CORRELATION_LAYER_HPP_
