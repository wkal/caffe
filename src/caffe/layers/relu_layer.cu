// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : 0;
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::NeuronForward_gpu(const Blob<Dtype>& bottom,
    Blob<Dtype>* top) {
  const Dtype* bottom_data = bottom.gpu_data();
  Dtype* top_data = top->mutable_gpu_data();
  const int count = bottom.count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (in_data[index] > 0);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::NeuronBackward_gpu(const Blob<Dtype>& top,
    Blob<Dtype>* bottom) {
  const Dtype* bottom_data = bottom->gpu_data();
  const Dtype* top_diff = top.gpu_diff();
  Dtype* bottom_diff = bottom->mutable_gpu_diff();
  const int count = bottom->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, bottom_data, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
