// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void SigmoidLayer<Dtype>::NeuronForward_cpu(const Blob<Dtype>& bottom,
    Blob<Dtype>* top) {
  const Dtype* bottom_data = bottom.cpu_data();
  Dtype* top_data = top->mutable_cpu_data();
  const int count = bottom.count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::NeuronBackward_cpu(const Blob<Dtype>& top,
    Blob<Dtype>* bottom) {
  const Dtype* top_data = top.cpu_data();
  const Dtype* top_diff = top.cpu_diff();
  Dtype* bottom_diff = bottom->mutable_cpu_diff();
  const int count = bottom->count();
  for (int i = 0; i < count; ++i) {
    const Dtype sigmoid_x = top_data[i];
    bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
  }
}

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
