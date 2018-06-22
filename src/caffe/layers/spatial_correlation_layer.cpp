#include <algorithm>
#include <vector>
#include <cmath>
#include "caffe/filler.hpp"
#include "caffe/layers/spatial_correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SpatialCorrelationLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(bottom.size(), 1) << "SpatialCorrelation layer only receive at most one blob as input";
}

template <typename Dtype>
void SpatialCorrelationLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(bottom[0]->channels());
  top_shape.push_back(2*(bottom[0]->height()));
  top_shape.push_back(2*(bottom[0]->width()));
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SpatialCorrelationLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SpatialCorrelationLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SpatialCorrelationLayer);
#endif

INSTANTIATE_CLASS(SpatialCorrelationLayer);
REGISTER_LAYER_CLASS(SpatialCorrelation);

}  // namespace caffe
