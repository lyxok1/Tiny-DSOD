#include <algorithm>
#include <vector>
#include <cmath>
#include "caffe/filler.hpp"
#include "caffe/layers/correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CorrelationLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CorrelationParameter corr_param = this->layer_param_.correlation_param();
  
  CHECK(corr_param.has_displacement()) << "Displacement must be specified for correlation";
  CHECK_LE(bottom.size(), 2) << "Correlation layer only receive at most two blobs as input";
  if(bottom.size()==2)
  {
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    self_ = false;
  }
  else
    self_ = true;

  kernel_size_ = corr_param.kernel_size();
  displacement_ = corr_param.displacement();
  if(corr_param.has_dilation())
    dilation_ = corr_param.dilation();
  else
    dilation_ = 1;

  if(!self_)
  {
    const int b0_w = bottom[0]->width();
    const int b1_w = bottom[1]->width();
    const int b0_h = bottom[0]->height();
    const int b1_h = bottom[1]->height();

    if(b0_w==b1_w)
      step_w_ = 1;
    else
      step_w_ = Dtype(b0_w)/b1_w;

    if(b0_h==b1_h)
      step_h_ = 1;
    else
      step_h_ = Dtype(b0_h)/b1_h; // get the length ratio of two blobs
  }
  else
  {
    step_w_ = 1;
    step_h_ = 1;
  }
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  const int window_size = displacement_*2+1;
  int idx = 0;
  if(!self_)
    idx = 1;
  top_shape.push_back(bottom[idx]->num());
  top_shape.push_back(window_size*window_size);
  top_shape.push_back(bottom[idx]->height());
  top_shape.push_back(bottom[idx]->width());
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if(self_)
  {
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int bottom_height = bottom[0]->height();
    const int bottom_width = bottom[0]->width();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int b0_dim = bottom[0]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int window_size = 2*displacement_+1;
    const int top_width = bottom_width;
    const int top_height = bottom_height;
    const int top_spatial = b0_spatial;
    const int top_dim = top_spatial*window_size*window_size;
    const int ks = kernel_size_/2;
    const int normalizer = kernel_size_*kernel_size_*channels;

    // correlation in kernel wise
    for(int n=0; n<num; n++)
      for(int x=0; x<top_width; x++)
        for(int y=0; y<top_height; y++)
        {
          const int b0_x = x;
          const int b0_y = y;
          for(int dx=-1*displacement_; dx<=displacement_; dx++)
            for(int dy=-1*displacement_; dy<=displacement_; dy++)
            {
              const int offset_x = b0_x + dilation_*dx;
              const int offset_y = b0_y + dilation_*dy;
              Dtype temp = 0.0;
              for(int kx=-1*ks; kx<=ks; kx++)
                for(int ky=-1*ks; ky<=ks; ky++)
                {
                  const int qx = offset_x + kx;
                  const int qy = offset_y + ky;
                  const int px = x + kx;
                  const int py = y + ky;
                  if(qx < bottom_width && qx >=0 && qy < bottom_height && qy >= 0 && 
                    px < bottom_width && px >=0 && py < bottom_height && py >= 0)
                  {
                    for(int c=0; c<channels;c++)
                      temp += bottom_data[n*b0_dim+c*b0_spatial+qy*bottom_width+qx]
                              *bottom_data[n*b0_dim+c*b0_spatial+py*bottom_width+px];
                  }
                }
              temp /= normalizer;
              const int out_c = (dy + displacement_)*window_size+(dx + displacement_);
              top_data[n*top_dim+out_c*top_spatial+y*top_width+x] = temp;
            }
        }
  }
  else
  {
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int bottom0_height = bottom[0]->height();
    const int bottom0_width = bottom[0]->width();
    const int bottom1_height = bottom[1]->height();
    const int bottom1_width = bottom[1]->width();
    const Dtype* bottom0_data = bottom[0]->cpu_data();
    const Dtype* bottom1_data = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int b0_dim = bottom[0]->count(1);
    const int b1_dim = bottom[1]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int b1_spatial = bottom[1]->count(2);
    const int window_size = 2*displacement_+1;
    const int top_width = bottom1_width;
    const int top_spatial = b1_spatial;
    const int top_dim = top_spatial*window_size*window_size;
    const int ks = kernel_size_/2;
    const int normalizer = kernel_size_*kernel_size_*channels;

    // correlation in kernel wise
    for(int n=0; n<num; n++)
      for(int x=0; x<bottom1_width; x++)
        for(int y=0; y<bottom1_height; y++)
        {
          const int b0_x = x*step_w_;
          const int b0_y = y*step_h_;
          for(int dx=-1*displacement_; dx<=displacement_; dx++)
            for(int dy=-1*displacement_; dy<=displacement_; dy++)
            {
              const int offset_x = b0_x + dilation_*dx;
              const int offset_y = b0_y + dilation_*dy;
              Dtype temp = 0.0;
              for(int kx=-1*ks; kx<=ks; kx++)
                for(int ky=-1*ks; ky<=ks; ky++)
                {
                  const int qx = offset_x + kx;
                  const int qy = offset_y + ky;
                  const int px = x + kx;
                  const int py = y + ky;
                  if(qx < bottom0_width && qx >=0 && qy < bottom0_height && qy >= 0 && 
                    px < bottom1_width && px >=0 && py < bottom1_height && py >= 0)
                  {
                    for(int c=0; c<channels;c++)
                      temp += bottom0_data[n*b0_dim+c*b0_spatial+qy*bottom0_width+qx]
                              *bottom1_data[n*b1_dim+c*b1_spatial+py*bottom1_width+px];
                  }
                }
              temp /= normalizer;
              const int out_c = (dy + displacement_)*window_size+(dx + displacement_);
              top_data[n*top_dim+out_c*top_spatial+y*top_width+x] = temp;
            }
        }
  }

}

template <typename Dtype>
void CorrelationLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  if(self_)
  {
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int bottom_height = bottom[0]->height();
    const int bottom_width = bottom[0]->width();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int b0_dim = bottom[0]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int window_size = 2*displacement_+1;
    const int top_width = bottom_width;
    const int top_spatial = b0_spatial;
    const int top_dim = top_spatial*window_size*window_size;
    const int ks = kernel_size_/2;
    const int normalizer = kernel_size_*kernel_size_*channels;

    if(propagate_down[0])
    {
      for(int n=0; n<num; n++)
        for(int c=0; c<channels; c++)
          for(int y=0; y<bottom_height; y++)
            for(int x=0; x<bottom_width; x++)
            {
              Dtype temp = 0.0;
              const int b0_x = x;
              const int b0_y = y;
              for(int dx=-1*displacement_; dx<=displacement_; dx++)
                for(int dy=-1*displacement_; dy<=displacement_; dy++)
                {
                  const int offset_x = b0_x + dilation_*dx;
                  const int offset_y = b0_y + dilation_*dy;
                  const int out_c = (dy + displacement_)*window_size+(dx + displacement_);

                  for(int kx=-1*ks;kx<=ks;kx++)
                    for(int ky=-1*ks;ky<=ks;ky++)
                    {
                      const int px = x - kx;
                      const int py = y - ky;
                      if(px>=0 && px<bottom_width && py>=0 && py<bottom_height)
                      {
                        if(offset_x == x && offset_y==y)
                        {
                            temp += 2*bottom_data[n*b0_dim+c*b0_spatial+y*bottom_width+x]
                                    *top_diff[n*top_dim+out_c*top_spatial+py*top_width+px];
                        }
                        else if(offset_x < bottom_width && offset_x >=0 && offset_y < bottom_height && offset_y >= 0)
                        {
                            temp += bottom_data[n*b0_dim+c*b0_spatial+offset_y*bottom_width+offset_x]
                                    *top_diff[n*top_dim+out_c*top_spatial+py*top_width+px];
                        }
                      }
                    }
                }

              for(int kx=-1*ks;kx<=ks;kx++)
                for(int ky=-1*ks;ky<=ks;ky++)
                {
                  const int px = x - kx;
                  const int py = y - ky;
                  for(int dx=-1*displacement_; dx<=displacement_; dx++)
                    for(int dy=-1*displacement_; dy<=displacement_; dy++)
                    {
                      if (dx==0 && dy==0)
                        continue;
                      const int offset_x = px + dilation_*dx;
                      const int offset_y = py + dilation_*dy;
                      const int qx = offset_x + kx;
                      const int qy = offset_y + ky;
                      const int out_c = (displacement_ - dy)*window_size+(displacement_ - dx);
                      if(offset_x < bottom_width && offset_x >=0 && offset_y < bottom_height && offset_y >= 0 &&
                        qx < bottom_width && qx >=0 && qy <bottom_height && qy >= 0)
                      {
                          temp += bottom_data[n*b0_dim+c*b0_spatial+qy*bottom_width+qx]
                                  *top_diff[n*top_dim+out_c*top_spatial+offset_y*top_width+offset_x];
                      }
                    }
                  }
              bottom_diff[n*b0_dim+c*b0_spatial+y*bottom_width+x] = temp/normalizer;
            }
    }
  }
  else
  {
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int bottom0_height = bottom[0]->height();
    const int bottom0_width = bottom[0]->width();
    const int bottom1_height = bottom[1]->height();
    const int bottom1_width = bottom[1]->width();
    const Dtype* bottom0_data = bottom[0]->cpu_data();
    const Dtype* bottom1_data = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom0_diff = bottom[0]->mutable_cpu_diff();
    Dtype* bottom1_diff = bottom[1]->mutable_cpu_diff();
    const int b0_dim = bottom[0]->count(1);
    const int b1_dim = bottom[1]->count(1);
    const int b0_spatial = bottom[0]->count(2);
    const int b1_spatial = bottom[1]->count(2);
    const int window_size = 2*displacement_+1;
    const int top_width = bottom1_width;
    const int top_spatial = b1_spatial;
    const int top_dim = top_spatial*window_size*window_size;
    const int ks = kernel_size_/2;
    const int normalizer = kernel_size_*kernel_size_*channels;
    
    if(propagate_down[1])
    {// backpropagate gradient for blob1
      for(int n=0; n<num; n++)
        for(int c=0; c<channels; c++)
          for(int x=0; x<bottom1_width; x++)
            for(int y=0; y<bottom1_height; y++)
            {
              Dtype temp = 0.0;
              for(int kx=-1*ks;kx<=ks;kx++)
                for(int ky=-1*ks;ky<=ks;ky++)
                {
                  const int px = x - kx;
                  const int py = y - ky;
                  if(px<bottom1_width && px>=0 && py<bottom1_height && py>=0)
                  {
                    const int b0_x = px*step_w_;
                    const int b0_y = py*step_h_;
                    for(int dx=-1*displacement_; dx<=displacement_; dx++)
                      for(int dy=-1*displacement_; dy<=displacement_; dy++)
                      {
                        const int offset_x = b0_x + dilation_*dx;
                        const int offset_y = b0_y + dilation_*dy;
                        const int out_c = (dy + displacement_)*window_size+(dx + displacement_);
                        const int qx = offset_x + kx;
                        const int qy = offset_y + ky;
                        if(qx < bottom0_width && qx >=0 && qy < bottom0_height && qy >= 0)
                        {
                            temp += bottom0_data[n*b0_dim+c*b0_spatial+qy*bottom0_width+qx]
                                    *top_diff[n*top_dim+out_c*top_spatial+py*top_width+px];
                        }
                      }
                  }
                }
              bottom1_diff[n*b1_dim+c*b1_spatial+y*bottom1_width+x] = temp/channels;
            }
    }

    if(propagate_down[0])
    {// backpropagate gradient for blob0
      for(int n=0; n<num; n++)
        for(int c=0; c<channels; c++)
          for(int x=0; x<bottom0_width; x++)
            for(int y=0; y<bottom0_height; y++)
            {
              Dtype temp = 0.0;
              for(int kx=-1*ks;kx<=ks;kx++)
                for(int ky=-1*ks;ky<=ks;ky++)
                {
                  const int qx = x - kx;
                  const int qy = y - ky;
                  if(qx<bottom0_width && qx>=0 && qy<bottom0_height && qy>=0)
                  {
                    for(int dx=-1*displacement_; dx<=displacement_; dx++)
                      for(int dy=-1*displacement_; dy<=displacement_; dy++)
                      {
                        const int b0_offset_x = qx + dx*dilation_;
                        const int b0_offset_y = qy + dy*dilation_;
                        //if(b0_offset_x >=0 && b0_offset_x<bottom0_width && b0_offset_y>=0 && b0_offset_y<bottom0_height)
                        //{
                          if(b0_offset_x%int(step_w_)==0 && b0_offset_y%int(step_h_)==0)
                          {
                            const int b1_x = b0_offset_x/step_w_;
                            const int b1_y = b0_offset_y/step_h_;
                            const int out_c = (displacement_ - dy)*window_size+(displacement_ - dx);
                            const int px = b1_x + kx;
                            const int py = b1_y + ky;
                            if(px<bottom1_width && px>=0 && py<bottom1_height && py>=0)
                              temp += top_diff[n*top_dim+out_c*top_spatial+b1_y*top_width+b1_x]
                                      *bottom1_data[n*b1_dim+c*b1_spatial+py*bottom1_width+px];
                          }
                        //}
                      }
                  }
                }
              bottom0_diff[n*b0_dim+c*b0_spatial+y*bottom0_width+x] = temp/normalizer;
            }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CorrelationLayer);
#endif

INSTANTIATE_CLASS(CorrelationLayer);
REGISTER_LAYER_CLASS(Correlation);

}  // namespace caffe
