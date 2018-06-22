#ifndef CAFFE_MM_LAYER_HPP_
#define CAFFE_MM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Mutiply two matrix. (b0 * b1 or b0 * weight)
 */
template <typename Dtype>
class MMLayer : public Layer<Dtype> {
 public:
  explicit MMLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MM"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_output_;
  Dtype eps_;

  bool normalize_;
  // blobs for normalization, only used when normalize is true and bottom.size()==1
  Blob<Dtype> norm_weight_;
  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> norm_;
  Blob<Dtype> buffer_;

};

}  // namespace caffe

#endif  // CAFFE_FLATTEN_LAYER_HPP_
