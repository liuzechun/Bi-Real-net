#ifdef USE_CUDNN
#include <vector>

#include "caffe/ex_layers/binary_conv_train_layer.hpp"

namespace caffe {

#define sign(x) ((x)>=0?1:-1)
#define clip(x) ((x)>=-1&&(x)<=1?x:0)
__global__ void binary_sync_conv_groups() { }

template <typename Dtype>
__global__ void BinaryGPU_binarize(const int n, const int num, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype sum = 0;
    for (int coor = 0; coor < num; coor++) {
      sum += std::abs(in[index * num + coor]) / Dtype(num);
    }
    for (int coor = 0; coor < num; coor++) {
      out[index * num + coor] = sign(in[index * num + coor]) * sum;
    }
  }  
}

template <typename Dtype>
void BinaryConvolutionTrainLayer<Dtype>::binarizeGPUTo(const shared_ptr<Blob<Dtype> > weights, const shared_ptr<Blob<Dtype> > wb) {
  const int count = weights->num();
  const int div = weights->count() / count;
  BinaryGPU_binarize<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, div, weights->gpu_data(), wb->mutable_gpu_data() );
}

template <typename Dtype>
void BinaryConvolutionTrainLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  binarizeGPUTo(this->blobs_[0], W_b);
  copyGPUdataTo(this->blobs_[0], W_buffer);
  copyGPUdataTo(W_b, this->blobs_[0]);
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
#if CUDNN_VERSION_MIN(4, 0, 0)
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
#else
        CUDNN_CHECK(cudnnAddTensor(handle_[g], CUDNN_ADD_SAME_C,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
#endif
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    binary_sync_conv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void BinaryConvolutionTrainLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
         CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g)); 
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
         CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    binary_sync_conv_groups<<<1, 1>>>();
  }
  copyGPUdataTo(W_buffer, this->blobs_[0]);
}

template void BinaryConvolutionTrainLayer<float>::binarizeGPUTo(const shared_ptr<Blob<float> > weights, const shared_ptr<Blob<float> > wb);
template void BinaryConvolutionTrainLayer<double>::binarizeGPUTo(const shared_ptr<Blob<double> > weights, const shared_ptr<Blob<double> > wb);

INSTANTIATE_LAYER_GPU_FUNCS(BinaryConvolutionTrainLayer);

}  // namespace caffe
#else

#include "caffe/ex_layers/binary_conv_train_layer.hpp"

namespace caffe {

template <typename Dtype>
void BinaryConvolutionTrainLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void BinaryConvolutionTrainLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryConvolutionTrainLayer);

}
#endif
