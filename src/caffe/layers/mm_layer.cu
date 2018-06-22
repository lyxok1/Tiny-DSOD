#include <vector>

#include "caffe/layers/mm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
      const int num = bottom[0]->shape(0);
      const int M = bottom[0]->shape(1);
      const int K = bottom[0]->shape(2);
      const int N = top[0]->shape(2);

      const int dim0 = M*K;
      const int out_dim = M*N;
      const int dim1 = K*N;

      const Dtype* b0_data = bottom[0]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();

      if(bottom.size()==1)
      {
        if(normalize_)
        {
          const Dtype* weights = this->blobs_[0]->gpu_data();
          Dtype* norm_data = norm_.mutable_gpu_data();
          Dtype* buffer_data = buffer_.mutable_gpu_data();
          const Dtype* mutiplier_data = sum_multiplier_.gpu_data();
          Dtype* norm_weights = norm_weight_.mutable_gpu_data();
          // weight normalization
          caffe_gpu_set(norm_.count(),Dtype(eps_), norm_data);
          caffe_gpu_powx(K*N, weights,Dtype(2.0), buffer_data);
          caffe_gpu_gemv<Dtype>(CblasTrans, K, N, Dtype(1.0), buffer_data,
              mutiplier_data, Dtype(1.0), norm_data);
          // generate norms
          caffe_gpu_powx(N, norm_data, Dtype(0.5), norm_data);
          // broadcast the norm
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K, N,
                                1, Dtype(1), mutiplier_data, norm_data,
                                Dtype(0), buffer_data);
          caffe_gpu_div<Dtype>(K*N, weights, buffer_data, norm_weights);

          // do matrix multiply
          for (int n = 0; n < num; ++n) 
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
              (Dtype)1., b0_data + n*dim0, norm_weights, 
              (Dtype)0., top_data + n*out_dim);
        }
        else
        {
          const Dtype* weights = this->blobs_[0]->gpu_data();

          for (int n = 0; n < num; ++n) 
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
              (Dtype)1., b0_data + n*dim0, weights, 
              (Dtype)0., top_data + n*out_dim);
        }
      }
      else
      {
        const Dtype* b1_data = bottom[1]->gpu_data();

        for (int n = 0; n < num; ++n) 
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
            (Dtype)1., b0_data + n*dim0, b1_data + n*dim1, 
            (Dtype)0., top_data + n*out_dim);
      }

  }

template <typename Dtype>
void MMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      const int num = bottom[0]->shape(0);
      const int M = bottom[0]->shape(1);
      const int K = bottom[0]->shape(2);
      const int N = top[0]->shape(2);

      const int dim0 = M*K;
      const int out_dim = M*N;
      const int dim1 = K*N;

      const Dtype* b0_data = bottom[0]->gpu_data();
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* b0_diff = bottom[0]->mutable_gpu_diff();

      if(bottom.size()==1)
      {
        if(!normalize_)
        {
          const Dtype* weights = this->blobs_[0]->gpu_data();
          Dtype* weights_diff = this->blobs_[0]->mutable_gpu_diff();

          if(propagate_down[0])
            for(int n=0; n< num; ++n)
              caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, K, N,
                (Dtype)1., top_diff + n*out_dim, weights,
                (Dtype)0., b0_diff + n*dim0);

          for (int n = 0; n < num; ++n)
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K, N, M,
              (Dtype)1., b0_data + n*dim0, top_diff + n*out_dim, 
              (Dtype)1., weights_diff);
        }
        else
        {
          // dW = 1/||W||(dw - (dw^T*w)w) 
          // [W is weights, w is norm weights, dw and dW are differentiated value]

          const Dtype* norm_weights = norm_weight_.gpu_data();
          Dtype* norm_weight_diff = norm_weight_.mutable_gpu_diff();
          Dtype* weights_diff = this->blobs_[0]->mutable_gpu_diff();
          const Dtype* mutiplier_data = sum_multiplier_.gpu_data();
          const Dtype* buffer_data = buffer_.gpu_data();

          // since this memory is never used, we use it to store
          // some intermediate data

          Dtype* mul_data = buffer_.mutable_gpu_diff();
          Dtype* coeff_data = norm_.mutable_gpu_diff();

          caffe_gpu_set(K*N, Dtype(0.0), norm_weight_diff);

          if(propagate_down[0])
            for(int n=0; n< num; ++n)
              caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, K, N,
                (Dtype)1., top_diff + n*out_dim, norm_weights,
                (Dtype)0., b0_diff + n*dim0);

          for (int n = 0; n < num; ++n)
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K, N, M,
              (Dtype)1., b0_data + n*dim0, top_diff + n*out_dim, 
              (Dtype)1., norm_weight_diff);

          caffe_gpu_mul(K*N, norm_weight_diff, norm_weights, mul_data);
          caffe_gpu_gemv(CblasTrans, K, N, Dtype(1.0), mul_data, mutiplier_data,
              Dtype(0.0), coeff_data);
          caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, K, N, 1,
              Dtype(1.0), mutiplier_data, coeff_data, Dtype(0.0), mul_data);
          caffe_gpu_mul(K*N, norm_weights, mul_data, mul_data);
          caffe_gpu_sub(K*N, norm_weight_diff, mul_data, mul_data);
          caffe_gpu_div(K*N, mul_data, buffer_data, mul_data);
          caffe_gpu_add(K*N, weights_diff, mul_data, weights_diff);
        }
      }
      else
      {
        const Dtype* b1_data = bottom[1]->gpu_data();
        Dtype* b1_diff = bottom[1]->mutable_gpu_diff();

        if(propagate_down[0])
          for (int n = 0; n < num; ++n)
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, K, N,
              (Dtype)1., top_diff + n*out_dim, b1_data + n*dim1,
              (Dtype)0., b0_diff + n*dim0);

        if(propagate_down[1])
          for (int n = 0; n < num; ++n)
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K, N, M,
              (Dtype)1., b0_data + n*dim0, top_diff + n*out_dim, 
              (Dtype)0., b1_diff + n*dim1);
      }
}

INSTANTIATE_LAYER_GPU_FUNCS(MMLayer);

}  // namespace caffe
