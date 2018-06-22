#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <time.h>

#include "caffe/layer_factory.hpp"
#include "caffe/layers/random_shuffle_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RandomShuffleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const RandomShuffleParameter& param = this->layer_param_.random_shuffle_param();
  axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(axis_,1) << "crossing-blobs random shuffle is not implemented";
  CHECK_LE(axis_,3) << "too many dimensions";
  if(this->blobs_.size()==1)
  {
    LOG(INFO) << "Skipping parameter initialization";
  }
  else if(this->blobs_.size()==0)
  {
    this->blobs_.resize(1);
    const int rn_num = bottom[0]->shape(axis_);
    vector<int> rn_shape(1,rn_num);
    this->blobs_[0].reset(new Blob<Dtype>(rn_shape));
    Dtype* rn_data = this->blobs_[0]->mutable_cpu_data();
    Dtype* rn_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(),Dtype(0.0),rn_diff);
    Dtype temp;
    for(int i=0;i<rn_num;i++)
      rn_data[i]=static_cast<Dtype>(i);
    // generate a random serial inside blobs_[0]
    srand((unsigned int)time(NULL));
    for(int i=1;i<rn_num;i++)
    {
      int x = rand()%i;
      temp = rn_data[i];
      rn_data[i] = rn_data[x];
      rn_data[x] = temp;
    }

  }
  else
    LOG(FATAL) << "too many inside blob parameters";
}

template <typename Dtype>
void RandomShuffleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for(int i=0;i<bottom.size();i++)
    top[i]->ReshapeLike(*bottom[i]);
}

template <typename Dtype>
void RandomShuffleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for(int k=0;k<bottom.size();k++)
  {
    const Dtype* bottom_data = bottom[k]->cpu_data();
    const Dtype* rn_data = this->blobs_[k]->cpu_data();
    Dtype* top_data = top[k]->mutable_cpu_data();
    const int inner_dim = bottom[k]->count(2);
    const int blob_dim = bottom[k]->count(1);
    const int width = bottom[k]->width();
    if(axis_==1)
    {
      for(int n=0;n<bottom[k]->num();n++)
        for(int i=0;i<this->blobs_[0]->shape(0);++i)
        {
          const int index = static_cast<int>(rn_data[i]);
          caffe_copy(inner_dim,bottom_data+n*blob_dim+i*inner_dim,top_data+n*blob_dim+index*inner_dim);
        }
    }
    else if(axis_==2)
    {
      for(int n=0;n<bottom[k]->num();n++)
        for(int i=0;i<this->blobs_[0]->shape(0);++i)
        {
          const int index = static_cast<int>(rn_data[i]);
          for(int ch=0;ch<bottom[k]->channels();ch++)
            for(int w=0;w<width;w++)
              top_data[n*blob_dim+ch*inner_dim+index*width+w] = bottom_data[n*blob_dim+ch*inner_dim+i*width+w];
        }
    }
    else if(axis_==3)
    {
      for(int n=0;n<bottom[k]->num();n++)
        for(int i=0;i<this->blobs_[0]->shape(0);++i)
        {
          const int index = static_cast<int>(rn_data[i]);
          for(int ch=0;ch<bottom[k]->channels();ch++)
            for(int h=0;h<bottom[k]->height();h++)
              top_data[n*blob_dim+ch*inner_dim+h*width+index] = bottom_data[n*blob_dim+ch*inner_dim+h*width+i];
        }
    }
  }
}

template <typename Dtype>
void RandomShuffleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for(int k=0;k<bottom.size();k++)
  {
    if(propagate_down[k])
    {
    	Dtype* bottom_diff = bottom[k]->mutable_cpu_diff();
    	const Dtype* rn_data = this->blobs_[k]->cpu_data();
    	const Dtype* top_diff = top[k]->cpu_diff();
    	const int inner_dim = bottom[k]->count(2);
    	const int blob_dim = bottom[k]->count(1);
    	const int width = bottom[k]->width();
    	if(axis_==1)
    	{
      		for(int n=0;n<bottom[k]->num();n++)
        		for(int i=0;i<this->blobs_[0]->shape(0);++i)
        		{
          			const int index = static_cast<int>(rn_data[i]);
          			caffe_copy(inner_dim,top_diff+n*blob_dim+index*inner_dim,bottom_diff+n*blob_dim+i*inner_dim);
        		}
    	}
    	else if(axis_==2)
    	{
      		for(int n=0;n<bottom[k]->num();n++)
        		for(int i=0;i<this->blobs_[0]->shape(0);++i)
        		{
          			const int index = static_cast<int>(rn_data[i]);
          			for(int ch=0;ch<bottom[k]->channels();ch++)
            				for(int w=0;w<width;w++)
              					bottom_diff[n*blob_dim+ch*inner_dim+i*width+w] = top_diff[n*blob_dim+ch*inner_dim+index*width+w];
        		}
    	}
    	else if(axis_==3)
    	{
      		for(int n=0;n<bottom[k]->num();n++)
        		for(int i=0;i<this->blobs_[0]->shape(0);++i)
        		{
          			const int index = static_cast<int>(rn_data[i]);
          			for(int ch=0;ch<bottom[k]->channels();ch++)
            				for(int h=0;h<bottom[k]->height();h++)
              					bottom_diff[n*blob_dim+ch*inner_dim+h*width+i] = top_diff[n*blob_dim+ch*inner_dim+h*width+index];
        		}
    	}
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(RandomShuffleLayer);
#endif

INSTANTIATE_CLASS(RandomShuffleLayer);
REGISTER_LAYER_CLASS(RandomShuffle);

}  // namespace caffe
