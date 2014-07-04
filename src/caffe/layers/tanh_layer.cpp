// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TanHLayer<Dtype>::NeuronForward_cpu(const Blob<Dtype>& bottom,
    Blob<Dtype>* top) {
  const Dtype* bottom_data = bottom.cpu_data();
  Dtype* top_data = top->mutable_cpu_data();
  Dtype exp2x;
  const int count = bottom.count();
  for (int i = 0; i < count; ++i) {
    exp2x = exp(2 * bottom_data[i]);
    top_data[i] = (exp2x - Dtype(1)) / (exp2x + Dtype(1));
  }
}

template <typename Dtype>
void TanHLayer<Dtype>::NeuronBackward_cpu(const Blob<Dtype>& top,
    Blob<Dtype>* bottom) {
  const Dtype* top_data = top.cpu_data();
  const Dtype* top_diff = top.cpu_diff();
  Dtype* bottom_diff = bottom->mutable_cpu_diff();
  const int count = bottom->count();
  Dtype tanhx;
  for (int i = 0; i < count; ++i) {
    tanhx = top_data[i];
    bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
  }
}

INSTANTIATE_CLASS(TanHLayer);

}  // namespace caffe
