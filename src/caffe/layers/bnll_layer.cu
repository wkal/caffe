// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

template <typename Dtype>
__global__ void BNLLForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ?
        in[index] + log(1. + exp(-in[index])) :
        log(1. + exp(in[index]));
  }
}

template <typename Dtype>
void BNLLLayer<Dtype>::NeuronForward_gpu(const Blob<Dtype>& bottom,
    Blob<Dtype>* top) {
  const Dtype* bottom_data = bottom.gpu_data();
  Dtype* top_data = top->mutable_gpu_data();
  const int count = bottom.count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  BNLLForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void BNLLBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype expval = exp(min(in_data[index], Dtype(kBNLL_THRESHOLD)));
    out_diff[index] = in_diff[index] * expval / (expval + 1.);
  }
}

template <typename Dtype>
void BNLLLayer<Dtype>::NeuronBackward_gpu(const Blob<Dtype>& top,
    Blob<Dtype>* bottom) {
  const Dtype* bottom_data = bottom->gpu_data();
  const Dtype* top_diff = top.gpu_diff();
  Dtype* bottom_diff = bottom->mutable_gpu_diff();
  const int count = bottom->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  BNLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, bottom_data, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_CLASS(BNLLLayer);


}  // namespace caffe
