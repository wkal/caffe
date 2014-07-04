// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::NeuronForward_cpu(const Blob<Dtype>& bottom,
    Blob<Dtype>* top) {
  const Dtype* bottom_data = bottom.cpu_data();
  Dtype* top_data = top->mutable_cpu_data();
  const int count = bottom.count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = max(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::NeuronBackward_cpu(const Blob<Dtype>& top,
    Blob<Dtype>* bottom) {
  const Dtype* bottom_data = bottom->cpu_data();
  const Dtype* top_diff = top.cpu_diff();
  Dtype* bottom_diff = bottom->mutable_cpu_diff();
  const int count = bottom->count();
  for (int i = 0; i < count; ++i) {
    bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0);
  }
}


INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
