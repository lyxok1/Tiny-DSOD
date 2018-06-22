#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/class_label_layer.hpp"

namespace caffe {

template <typename Dtype>
void ClassLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ClassLabelParameter& class_label_param =
      this->layer_param_.class_label_param();
  label_type_ = class_label_param.label_type();

  if(label_type_ == ClassLabelParameter_Label_type_CLASSIFICATION)
  {
    CHECK(class_label_param.has_num_classes()) << "number of classes must be specified.";
    num_classes_ = class_label_param.num_classes();
  }
  else if(label_type_ == ClassLabelParameter_Label_type_SEGMENTATION)
  {
    variance_ = class_label_param.variance();
  }
  else
    LOG(FATAL) << "No such label type";
}

template <typename Dtype>
void ClassLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = bottom[0]->num();
  const int width = bottom[0]->width();
  const int height = bottom[0]->height();
  const int channels = num_classes_;
  
  if(label_type_ == ClassLabelParameter_Label_type_CLASSIFICATION)
  {
    vector<int> top_shape(4, 1);
    top_shape[0] = batch_size;
    top_shape[1] = channels;
    top[0]->Reshape(top_shape);
  }
  else{
    vector<int> top_shape(4, 1);
    top_shape[0] = batch_size;
    top_shape[2] = height;
    top_shape[3] = width;
    top[0]->Reshape(top_shape);
  }

}

template <typename Dtype>
void ClassLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // for each annotated label, there are 8 floating point numbers to indicates the 
  // label information: size [1, 1, c, 8]
  //   [item_id, class label, instance id, xmin, ymin, xmax, ymax, difficulty]
  // here we only extract item_id to indicate the index of each image
  // and the class label to show what classes are included inside a image for classification
  // 
  // for segmentation weak supervision, we extract xmin, ymin, xmax, ymax for location

  const int label_dim = bottom[1]->shape(3); // 8
  Dtype* top_data = top[0]->mutable_cpu_data();

  const Dtype* anno_data = bottom[1]->cpu_data();
  const int anno_num = bottom[1]->shape(2); // c

  const int top_dim = top[0]->count(1);
  const int width = bottom[0]->width();
  const int height = bottom[0]->height();

  caffe_set(top[0]->count(), Dtype(0.0), top_data);

  int item_id = 0;
  int class_id = 0;
  int wstart = 0;
  int hstart = 0;
  int wend = 0;
  int hend = 0;

  Dtype xmin = 0.0;
  Dtype xmax = 0.0;
  Dtype ymin = 0.0;
  Dtype ymax = 0.0;
  Dtype wmean = 0.0;
  Dtype hmean = 0.0;
  Dtype dist = 0.0;

  if(label_type_ == ClassLabelParameter_Label_type_CLASSIFICATION)
  {
    // generate classification label
    for(int a=0; a<anno_num; a++)
    {
      item_id = anno_data[a*label_dim];
      class_id = anno_data[a*label_dim + 1];

      // the corresponding position in output label is set 1
      top_data[item_id*top_dim + class_id] = Dtype(1.);
    }
  }
  else
  {
     // generate segmentation weak labels
     caffe_set(top[0]->count(), Dtype(0.0), top_data);
     
     for(int a=0; a<anno_num; a++)
     {
        item_id = anno_data[a*label_dim];
        xmin = anno_data[a*label_dim + 3]*width;
        xmax = anno_data[a*label_dim + 5]*width;
        ymin = anno_data[a*label_dim + 4]*height;
        ymax = anno_data[a*label_dim + 6]*height;
        wstart = std::max(int(xmin), 0);
        wend = std::min(int(xmax), width - 1);
        hstart = std::max(int(ymin), 0);
        hend = std::min(int(ymax), height - 1);

        wmean = (xmin + xmax) / 2;
        hmean = (ymin + ymax) / 2;
        // the corresponding position in output label is set 1
        for(int x=wstart; x<=wend; x++)
          for(int y=hstart; y<=hend; y++)
          {
            dist = pow(x - wmean, 2) + pow(y - hmean, 2);
            Dtype temp = exp(-1*dist/variance_);

            top_data[item_id*top_dim + y*width + x] = std::max(top_data[item_id*top_dim + y*width + x], temp);
          }
     }
  }
}

INSTANTIATE_CLASS(ClassLabelLayer);
REGISTER_LAYER_CLASS(ClassLabel);

}  // namespace caffe
