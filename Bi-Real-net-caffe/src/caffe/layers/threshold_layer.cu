#include <vector>

#include "caffe/layers/threshold_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ThresholdForward(const int n, const Dtype threshold,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? 1 : -1;
  }
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ThresholdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, threshold_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
__global__ void ThresholdBackward(const int count, const Dtype* bottom_data, const Dtype* top_diff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, count) {
    if (bottom_data[index] >= Dtype(-1) && bottom_data[index] < Dtype(0)) {
      bottom_diff[ index ] = (2+2*bottom_data[index])*top_diff[ index ];
    } else if (bottom_data[index] >= Dtype(0) && bottom_data[index] < Dtype(1)) {
      bottom_diff[ index ] = (2-2*bottom_data[index])*top_diff[ index ];
    } else {
      bottom_diff[ index ] = Dtype(0);
    }
  }
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if ( propagate_down[0] == false ) return;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = top[0]->count();
  ThresholdBackward<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(ThresholdLayer);

}  // namespace caffe
