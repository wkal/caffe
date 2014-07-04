// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::min;

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

template <typename Dtype>
void BNLLLayer<Dtype>::NeuronForward_cpu(const Blob<Dtype>& bottom,
    Blob<Dtype>* top) {
  const Dtype* bottom_data = bottom.cpu_data();
  Dtype* top_data = top->mutable_cpu_data();
  const int count = bottom.count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] > 0 ?
        bottom_data[i] + log(1. + exp(-bottom_data[i])) :
        log(1. + exp(bottom_data[i]));
  }
}

template <typename Dtype>
void BNLLLayer<Dtype>::NeuronBackward_cpu(const Blob<Dtype>& top,
    Blob<Dtype>* bottom) {
  const Dtype* bottom_data = bottom->cpu_data();
  const Dtype* top_diff = top.cpu_diff();
  Dtype* bottom_diff = bottom->mutable_cpu_diff();
  const int count = bottom->count();
  Dtype expval;
  for (int i = 0; i < count; ++i) {
    expval = exp(min(bottom_data[i], Dtype(kBNLL_THRESHOLD)));
    bottom_diff[i] = top_diff[i] * expval / (expval + 1.);
  }
}


INSTANTIATE_CLASS(BNLLLayer);


}  // namespace caffe
