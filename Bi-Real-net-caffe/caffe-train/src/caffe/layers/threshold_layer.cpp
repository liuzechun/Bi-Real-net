#include <vector>

#include "caffe/layers/threshold_layer.hpp"

namespace caffe {

template <typename Dtype>
void ThresholdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.threshold_param().threshold();
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] > threshold_) ? Dtype(1) : Dtype(-1);
  }
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if ( propagate_down[0] == false ) return;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int index = 0; index < top[0]->count(); index++) {
    if ( bottom_data[index] >= Dtype(-1) && bottom_data[index] < Dtype(0) ) {
      bottom_diff[ index ] = (2+2*bottom_data[index])*top_diff[ index ];
    } else if ( bottom_data[index] >= Dtype(0) && bottom_data[index] < Dtype(1) ) {
      bottom_diff[ index ] = (2-2*bottom_data[index])*top_diff[ index ];
    } else {
      bottom_diff[ index ] = Dtype(0);
    }
  } 
}

#ifdef CPU_ONLY
STUB_GPU(ThresholdLayer);
#endif

INSTANTIATE_CLASS(ThresholdLayer);
REGISTER_LAYER_CLASS(Threshold);

}  // namespace caffe
