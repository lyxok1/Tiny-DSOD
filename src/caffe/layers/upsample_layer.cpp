#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "caffe/filler.hpp"
#include "caffe/layers/upsample_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void UpsampleLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CorrelationParameter corr_param = this->layer_param_.correlation_param();
  
  CHECK_EQ(bottom.size(), 2) << "upsample layer receive exactly two blobs";
  const int b0_width = bottom[0]->width();
  const int b0_height = bottom[0]->height();
  const int b1_width = bottom[1]->width();
  const int b1_height = bottom[1]->height();

  CHECK_GE(b0_width, b1_width);
  CHECK_GE(b0_height, b1_height);

  w_ratio_ = Dtype(b0_width)/b1_width;
  h_ratio_ = Dtype(b0_height)/b1_height;

  vector<int> container_shape;
  container_shape.push_back(b0_height);
  container_shape.push_back(b0_width);
  container_shape.push_back(b1_height);
  container_shape.push_back(b1_width);
  container_.Reshape(container_shape);
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int b0_height = bottom[0]->height();
    const int b0_width = bottom[0]->width();
    const int b1_height = bottom[1]->height();
    const int b1_width = bottom[1]->width();
    const Dtype* b1_data = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int b0_dim = bottom[0]->count(1);
    const int b1_dim = bottom[1]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int b1_spatial = bottom[1]->count(2);

    // correlation in kernel wise
    for(int n=0; n<num; n++)
      for(int x=0; x<b0_width; x++)
        for(int y=0; y<b0_height; y++)
        {
          const Dtype b1_x = (x+1)/w_ratio_ - 1.0;
          const Dtype b1_y = (y+1)/h_ratio_ - 1.0;
          
          const int x_l = std::max(int(b1_x), 0);
          const int x_h = std::min(int(b1_x)+1, b1_width-1);
          const Dtype wxl = int(b1_x) < 0 ? 0.0 : (1 - (b1_x - int(b1_x)));
          const Dtype wxh = int(b1_x) + 1 > (b1_width-1) ? 0.0 : ( 1 - wxl);

          const int y_l = std::max(int(b1_y), 0);
          const int y_h = std::min(int(b1_y)+1, b1_height-1);
          const Dtype wyl = int(b1_y) < 0 ? 0.0 : (1 - (b1_y - int(b1_y)));
          const Dtype wyh = int(b1_y) + 1 > (b1_height-1) ? 0.0 : ( 1 - wyl);

          for(int c=0; c<channels ;c++)
          {
            Dtype temp = 0.0;
            temp += wxl*wyl*b1_data[n*b1_dim+c*b1_spatial+y_l*b1_width+x_l];
            temp += wxh*wyl*b1_data[n*b1_dim+c*b1_spatial+y_l*b1_width+x_h];
            temp += wxl*wyh*b1_data[n*b1_dim+c*b1_spatial+y_h*b1_width+x_l];
            temp += wxh*wyh*b1_data[n*b1_dim+c*b1_spatial+y_h*b1_width+x_h];
            top_data[n*b0_dim+c*b0_spatial+y*b0_width+x] = temp;
          }
        }
}

template <typename Dtype> 
void UpsampleLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int b0_height = bottom[0]->height();
    const int b0_width = bottom[0]->width();
    const int b1_height = bottom[1]->height();
    const int b1_width = bottom[1]->width();
    const Dtype* b1_data = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* b1_diff = bottom[1]->mutable_cpu_diff();
    const int b0_dim = bottom[0]->count(1);
    const int b1_dim = bottom[1]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int b1_spatial = bottom[1]->count(2);
    
    if(propagate_down[1])
    {// backpropagate gradient for blob1
      caffe_set(bottom[1]->count(), Dtype(0.0), b1_diff);
      for(int n=0; n<num; n++)
          for(int x=0; x<b0_width; x++)
            for(int y=0; y<b0_height; y++)
            {
              const Dtype b1_x = (x+1)/w_ratio_ - 1.0;
              const Dtype b1_y = (y+1)/h_ratio_ - 1.0;
              
              const int x_l = std::max(int(b1_x), 0);
              const int x_h = std::min(int(b1_x)+1, b1_width-1);
              const Dtype wxl = int(b1_x) < 0 ? 0.0 : (1 - (b1_x - int(b1_x)));
              const Dtype wxh = int(b1_x) + 1 > (b1_width-1) ? 0.0 : ( 1 - wxl);

              const int y_l = std::max(int(b1_y), 0);
              const int y_h = std::min(int(b1_y)+1, b1_height-1);
              const Dtype wyl = int(b1_y) < 0 ? 0.0 : (1 - (b1_y - int(b1_y)));
              const Dtype wyh = int(b1_y) + 1 > (b1_height-1) ? 0.0 : ( 1 - wyl);

              for(int c=0; c<channels ;c++)
              {
                Dtype temp = 0.0;
                b1_diff[n*b1_dim+c*b1_spatial+y_l*b1_width+x_l] += wxl*wyl*top_diff[n*b0_dim+c*b0_spatial+y*b0_width+x];
                b1_diff[n*b1_dim+c*b1_spatial+y_l*b1_width+x_h] += wxh*wyl*top_diff[n*b0_dim+c*b0_spatial+y*b0_width+x];
                b1_diff[n*b1_dim+c*b1_spatial+y_h*b1_width+x_l] += wxl*wyh*top_diff[n*b0_dim+c*b0_spatial+y*b0_width+x];
                b1_diff[n*b1_dim+c*b1_spatial+y_h*b1_width+x_h] += wxh*wyh*top_diff[n*b0_dim+c*b0_spatial+y*b0_width+x];
              }
            }
    }

    if(propagate_down[0])
      NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(UpsampleLayer);
#endif

INSTANTIATE_CLASS(UpsampleLayer);
REGISTER_LAYER_CLASS(Upsample);

}  // namespace caffe
