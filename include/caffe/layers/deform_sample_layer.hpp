#ifndef CAFFE_DEFORM_SAMPLE_LAYER_HPP_
#define CAFFE_DEFORM_SAMPLE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

//TODO: implement multiple sample in channel wise

/**
 * @brief sample feature maps according to a series of given offsets (dx,dy)
          for each location on input feature map, we cut the channel wise into s samples,
          each sample is corresponded to an offsets tuple (dx, dy)
 * @input:
 *    blob0: input feature map for pooling [N x C x H x W]
 *    blob1: offset blobs for each sample center position [N x (2*num_samples_per_loc) x H x W]
 *    # requirement: the C and num_samples_per_loc meets: C % num_samples_per_loc == 0
 * @output:
 *    pooling: summation of sampled features in a 4-dimensional format [ N x C x H x W]
 *
 * created on: 2018-3-29
 * author: Li Yuxi
 */

template <typename Dtype>
class DeformSampleLayer : public Layer<Dtype> {
 public:
  explicit DeformSampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DeformSample"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
    NOT_IMPLEMENTED;
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_samples_per_loc_;
  // blob to contain bilinear coefficient
  Blob<Dtype> bi_coeff_;

};

}  // namespace caffe

#endif  // CAFFE_FLATTEN_LAYER_HPP_
