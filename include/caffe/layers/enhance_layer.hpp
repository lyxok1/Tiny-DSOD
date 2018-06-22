#ifndef CAFFE_ENHANCE_LAYER_HPP_
#define CAFFE_ENHANCE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/correlation_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype> class  CorrelationLayer;
template <typename Dtype> class  SoftmaxLayer;

/**
 * @brief aggregate other spatial local features to enhance local representation
 *        
 */
template <typename Dtype>
class EnhanceLayer : public Layer<Dtype> {
 public:
  explicit EnhanceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Enhance"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // tool layers for feature enhancement
  shared_ptr<CorrelationLayer<Dtype> > correlation_;
  Blob<Dtype> corr_;
  vector<Blob<Dtype>*> corr_top_vec_;
  shared_ptr<SoftmaxLayer<Dtype> > softmax_;
  Blob<Dtype> soft_;
  vector<Blob<Dtype>*> soft_top_vec_;

  int displacement_;
  int kernel_size_;
  int dilation_;

};

}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
