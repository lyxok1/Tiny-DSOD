#ifndef CAFFE_UPSAMPLE_LAYER_HPP_
#define CAFFE_UPSAMPLE_LAYER_HPP_

/*
* written in 2018.3.12
* Author Yuxi Li
*/

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"

namespace caffe {

// upsample layer:
// @brief: given two different blobs, upsample blob 1 to the size of blob 0
//         by bilinear interpolation

template <typename Dtype>
class UpsampleLayer : public Layer<Dtype> {
 public:
  explicit UpsampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 2;}
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "Upsample"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  Dtype w_ratio_;
  Dtype h_ratio_;
  Blob<Dtype> container_; // auxiliary container for backward gradients
};

}  // namespace caffe


#endif  // CAFFE_UPSAMPLE_LAYER_HPP_
