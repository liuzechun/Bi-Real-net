#include <algorithm>
#include <vector>

#include "caffe/ex_layers/leaky_clip_layer.hpp"

namespace caffe {

template <typename Dtype>
void lClipLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    if ( bottom_data[i] <= Dtype(-1) ) {
      top_data[i] = -1 + negative_slope * (bottom_data[i]+1);
    } else if ( bottom_data[i] > Dtype(-1) && bottom_data[i] < Dtype(1) ) {
      top_data[i] = bottom_data[i] ;
    } else {
      top_data[i] =  1 + negative_slope * (bottom_data[i]-1);
    }
  }
}

template <typename Dtype>
void lClipLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      if ( bottom_data[i] > Dtype(-1) && bottom_data[i] < Dtype(1) ) {
        bottom_diff[i] = top_diff[i];
      } else {
        bottom_diff[i] = negative_slope * top_diff[i];
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(lClipLayer);
#endif

INSTANTIATE_CLASS(lClipLayer);
REGISTER_LAYER_CLASS(lClip);

}  // namespace caffe
