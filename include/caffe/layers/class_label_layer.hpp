#ifndef CAFFE_CLASS_LABEL_LAYER_HPP_
#define CAFFE_CLASS_LABEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Extract the class label from annotated data blobs
 *        
 * Intended for use with sigmoid crossentropy loss.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class ClassLabelLayer : public Layer<Dtype> {
 public:
  /**
   * @param
   *        input blob 0 contain the score for each class (just to help indicate the batch_size)
   *        input blob 1 contain annotated labels, size [1, 1, c, 8]
   *        -num_classes: int, class numbers
   * @output
   *        output blob of size [n, c, 1, 1], each data is a indicate variable to 
   *        indicate the existance of certain class type
   */
  explicit ClassLabelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ClassLabel"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
  }

  int num_classes_;
  float variance_;
  ClassLabelParameter_Label_type label_type_;

};

}  // namespace caffe

#endif  // CAFFE_PRIORBOX_LAYER_HPP_
