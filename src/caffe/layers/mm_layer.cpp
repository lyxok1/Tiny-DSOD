#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/mm_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <time.h>

namespace caffe {

template <typename Dtype>
void MMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  if(bottom.size()==1)
  {
    // generate a learnable matrix
    CHECK(this->layer_param_.mm_param().has_num_output()) << "output number should be specified.";
    num_output_ = this->layer_param_.mm_param().num_output();
    normalize_ = this->layer_param_.mm_param().normalize();
    eps_ = this->layer_param_.mm_param().eps();

    this->blobs_.resize(1);
    // Initialize and fill the weights:
    vector<int> weight_shape;
    weight_shape.push_back(bottom[0]->shape(2));
    weight_shape.push_back(num_output_);

    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.mm_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }
  else if(bottom.size()==2)
  {
    CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(1)) << "Shape mismatch.";
  }
    
}

template <typename Dtype>
void MMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> output_shape;
  output_shape.push_back(bottom[0]->shape(0));
  output_shape.push_back(bottom[0]->shape(1));

  if(bottom.size()==1)
    output_shape.push_back(this->blobs_[0]->shape(1));
  else if(bottom.size()==2)
    output_shape.push_back(bottom[1]->shape(2));

  top[0]->Reshape(output_shape);

  if(normalize_)
  {
    norm_weight_.ReshapeLike(*(this->blobs_[0]));
    buffer_.ReshapeLike(*(this->blobs_[0]));
    norm_.Reshape(vector<int>(1, num_output_));
    sum_multiplier_.Reshape(vector<int>(1, bottom[0]->shape(2)));
    caffe_set(sum_multiplier_.count(), Dtype(1.0), sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void MMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
      const int num = bottom[0]->shape(0);
      const int M = bottom[0]->shape(1);
      const int K = bottom[0]->shape(2);
      const int N = top[0]->shape(2);

      const int dim0 = M*K;
      const int out_dim = M*N;
      const int dim1 = K*N;

      const Dtype* b0_data = bottom[0]->cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();

      if(bottom.size()==1)
      {
        if(normalize_)
        {
          const Dtype* weights = this->blobs_[0]->cpu_data();
          Dtype* norm_data = norm_.mutable_cpu_data();
          Dtype* buffer_data = buffer_.mutable_cpu_data();
          const Dtype* mutiplier_data = sum_multiplier_.cpu_data();
          Dtype* norm_weights = norm_weight_.mutable_cpu_data();
          // weight normalization
          caffe_set(norm_.count(),Dtype(eps_), norm_data);
          caffe_sqr(K*N, weights, buffer_data);
          caffe_cpu_gemv<Dtype>(CblasTrans, K, N, Dtype(1.0), buffer_data,
              mutiplier_data, Dtype(1.0), norm_data);
          // generate norms
          caffe_powx(N, norm_data, Dtype(0.5), norm_data);
          // broadcast the norm
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K, N,
                                1, Dtype(1), mutiplier_data, norm_data,
                                Dtype(0), buffer_data);
          caffe_div<Dtype>(K*N, weights, buffer_data, norm_weights);

          // do matrix multiply
          for (int n = 0; n < num; ++n) 
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
              (Dtype)1., b0_data + n*dim0, norm_weights, 
              (Dtype)0., top_data + n*out_dim);
        }
        else
        {
          const Dtype* weights = this->blobs_[0]->cpu_data();

          for (int n = 0; n < num; ++n) 
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
              (Dtype)1., b0_data + n*dim0, weights, 
              (Dtype)0., top_data + n*out_dim);
        }
      }
      else
      {
        const Dtype* b1_data = bottom[1]->cpu_data();

        for (int n = 0; n < num; ++n) 
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
            (Dtype)1., b0_data + n*dim0, b1_data + n*dim1, 
            (Dtype)0., top_data + n*out_dim);
      }

  }

template <typename Dtype>
void MMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      const int num = bottom[0]->shape(0);
      const int M = bottom[0]->shape(1);
      const int K = bottom[0]->shape(2);
      const int N = top[0]->shape(2);

      const int dim0 = M*K;
      const int out_dim = M*N;
      const int dim1 = K*N;

      const Dtype* b0_data = bottom[0]->cpu_data();
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* b0_diff = bottom[0]->mutable_cpu_diff();

      if(bottom.size()==1)
      {
        if(!normalize_)
        {
          const Dtype* weights = this->blobs_[0]->cpu_data();
          Dtype* weights_diff = this->blobs_[0]->mutable_cpu_diff();

          if(propagate_down[0])
            for(int n=0; n< num; ++n)
              caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, K, N,
                (Dtype)1., top_diff + n*out_dim, weights,
                (Dtype)0., b0_diff + n*dim0);

          for (int n = 0; n < num; ++n)
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K, N, M,
              (Dtype)1., b0_data + n*dim0, top_diff + n*out_dim, 
              (Dtype)1., weights_diff);
        }
        else
        {
          const Dtype* norm_weights = norm_weight_.cpu_data();
          Dtype* norm_weight_diff = norm_weight_.mutable_cpu_diff();
          Dtype* weights_diff = this->blobs_[0]->mutable_cpu_diff();
          const Dtype* mutiplier_data = sum_multiplier_.cpu_data();
          const Dtype* buffer_data = buffer_.cpu_data();

          // since this memory is never used, we use it to store
          // some intermediate data

          // dW = 1/||W||(dw - (dw^T*w)w) 
          // [W is weights, w is norm weights, dw and dW are differentiated value]

          Dtype* mul_data = buffer_.mutable_cpu_diff();
          Dtype* coeff_data = norm_.mutable_cpu_diff();

          caffe_set(K*N, Dtype(0.0), norm_weight_diff);

          if(propagate_down[0])
            for(int n=0; n< num; ++n)
              caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, K, N,
                (Dtype)1., top_diff + n*out_dim, norm_weights,
                (Dtype)0., b0_diff + n*dim0);

          for (int n = 0; n < num; ++n)
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K, N, M,
              (Dtype)1., b0_data + n*dim0, top_diff + n*out_dim, 
              (Dtype)1., norm_weight_diff);

          caffe_mul(K*N, norm_weight_diff, norm_weights, mul_data);
          caffe_cpu_gemv(CblasTrans, K, N, Dtype(1.0), mul_data, mutiplier_data,
              Dtype(0.0), coeff_data);
          caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, K, N, 1,
              Dtype(1.0), mutiplier_data, coeff_data, Dtype(0.0), mul_data);
          caffe_mul(K*N, norm_weights, mul_data, mul_data);
          caffe_sub(K*N, norm_weight_diff, mul_data, mul_data);
          caffe_div(K*N, mul_data, buffer_data, mul_data);
          caffe_add(K*N, weights_diff, mul_data, weights_diff);
        }
      }
      else
      {
        const Dtype* b1_data = bottom[1]->cpu_data();
        Dtype* b1_diff = bottom[1]->mutable_cpu_diff();

        if(propagate_down[0])
          for (int n = 0; n < num; ++n)
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, K, N,
              (Dtype)1., top_diff + n*out_dim, b1_data + n*dim1,
              (Dtype)0., b0_diff + n*dim0);

        if(propagate_down[1])
          for (int n = 0; n < num; ++n)
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K, N, M,
              (Dtype)1., b0_data + n*dim0, top_diff + n*out_dim, 
              (Dtype)0., b1_diff + n*dim1);
      }
}

#ifdef CPU_ONLY
STUB_GPU(MMLayer);
#endif

INSTANTIATE_CLASS(MMLayer);
REGISTER_LAYER_CLASS(MM);

}  // namespace caffe
