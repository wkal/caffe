// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void ThresholdLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  threshold_ = this->layer_param_.threshold_param().threshold();
}

template <typename Dtype>
void ThresholdLayer<Dtype>::NeuronForward_cpu(const Blob<Dtype>& bottom,
    Blob<Dtype>* top) {
  const Dtype* bottom_data = bottom.cpu_data();
  Dtype* top_data = top->mutable_cpu_data();
  const int count = bottom.count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] > threshold_) ? Dtype(1) : Dtype(0);
  }
}

INSTANTIATE_CLASS(ThresholdLayer);

}  // namespace caffe
