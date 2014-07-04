// Copyright 2014 BVLC and contributors.

// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.resize(bottom.size());
  for (int i = 0; i < bottom.size(); ++i) {
    rand_vec_[i].reset(new Blob<unsigned int>(bottom[i]->num(),
        bottom[i]->channels(), bottom[i]->height(), bottom[i]->width()));
  }
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
Dtype DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    unsigned int* mask = rand_vec_[i]->mutable_cpu_data();
    const int count = bottom[i]->count();
    if (Caffe::phase() == Caffe::TRAIN) {
      // Create random numbers
      caffe_rng_bernoulli(count, 1. - threshold_, mask);
      for (int i = 0; i < count; ++i) {
        top_data[i] = bottom_data[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(bottom[i]->count(), bottom_data, top_data);
    }
  }
  return Dtype(0);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* top_diff = top[i]->cpu_diff();
      Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
      if (Caffe::phase() == Caffe::TRAIN) {
        const unsigned int* mask = rand_vec_[i]->cpu_data();
        const int count = (*bottom)[i]->count();
        for (int j = 0; j < count; ++j) {
          bottom_diff[j] = top_diff[j] * mask[j] * scale_;
        }
      } else {
        caffe_copy(top[i]->count(), top_diff, bottom_diff);
      }
    }
  }
}


INSTANTIATE_CLASS(DropoutLayer);


}  // namespace caffe
