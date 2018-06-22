#include <algorithm>
#include <vector>

#include "caffe/layers/deform_sample_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <time.h>

namespace caffe {

template <typename Dtype>
void DeformSampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(),2);
  DeformSampleParameter deform_sample_param = this->layer_param_.deform_sample_param();
  CHECK(deform_sample_param.has_num_samples_per_loc());
  num_samples_per_loc_ = deform_sample_param.num_samples_per_loc();
  CHECK_EQ(bottom[1]->channels()/2, num_samples_per_loc_);
  CHECK_EQ(bottom[0]->channels()%num_samples_per_loc_, 0);
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());

  const int spatial = bottom[0]->count(2);
  vector<int> coeff_shape;

  coeff_shape.push_back(bottom[1]->num());
  coeff_shape.push_back(num_samples_per_loc_);
  coeff_shape.push_back(spatial);
  coeff_shape.push_back(spatial);
  bi_coeff_.Reshape(coeff_shape);
}

template <typename Dtype>
void DeformSampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

#ifdef CPU_ONLY
STUB_GPU(DeformSampleLayer);
#endif

INSTANTIATE_CLASS(DeformSampleLayer);
REGISTER_LAYER_CLASS(DeformSample);

}  // namespace caffe
