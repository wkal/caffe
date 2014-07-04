// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  // NeuronLayer allows in-place computations. If the computation is not
  // in-place, we will need to initialize the top blob.
  for (int i = 0; i < bottom.size(); ++i) {
    if ((*top)[i] != bottom[i]) {
      (*top)[i]->Reshape(bottom[i]->num(), bottom[i]->channels(),
          bottom[i]->height(), bottom[i]->width());
    }
  }
}

template <typename Dtype>
Dtype NeuronLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    this->NeuronForward_cpu(*bottom[i], (*top)[i]);
  }
  return Dtype(0);
}

template <typename Dtype>
void NeuronLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      this->NeuronBackward_cpu(*top[i], (*bottom)[i]);
    }
  }
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
