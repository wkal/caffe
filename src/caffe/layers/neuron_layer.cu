// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Dtype NeuronLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    this->NeuronForward_gpu(*bottom[i], (*top)[i]);
  }
  return Dtype(0);
}

template <typename Dtype>
void NeuronLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      this->NeuronBackward_gpu(*top[i], (*bottom)[i]);
    }
  }
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
