import unittest
import tempfile
import os

import caffe
from caffe.proto import caffe_pb2

class SimpleLayer(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].num, bottom[0].channels, bottom[0].height,
                bottom[0].width)

    def forward(self, bottom, top):
        top[0].data[...] = 10 * bottom[0].data

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = 10 * top[0].diff

def python_net_param():
    net_param = caffe_pb2.NetParameter()
    net_param.name = 'pythonnet'
    net_param.force_backward = True
    net_param.input.append('data')
    input_dims = [10, 9, 8, 7]
    for dim in input_dims: net_param.input_dim.append(dim)
    names = ['data', 'one', 'two', 'three']
    for input_name, name in zip(names[:-1], names[1:]):
        python_layer = net_param.layer.add()
        python_layer.name = name
        python_layer.type = 'Python'
        python_layer.bottom.append(input_name)
        python_layer.top.append(name)
        python_layer.python_param.module = 'test_python_layer'
        python_layer.python_param.layer = 'SimpleLayer'
    return net_param

class TestPythonLayer(unittest.TestCase):
    def setUp(self):
        net_param = python_net_param()
        self.net = caffe.Net(net_param.SerializeToString())

    def test_forward(self):
        x = 8
        self.net.blobs['data'].data[...] = x
        self.net.forward()
        for y in self.net.blobs['three'].data.flat:
            self.assertEqual(y, 10**3 * x)

    def test_backward(self):
        x = 7
        self.net.blobs['three'].diff[...] = x
        self.net.backward()
        for y in self.net.blobs['data'].diff.flat:
            self.assertEqual(y, 10**3 * x)

    def test_reshape(self):
        s = 4
        self.net.blobs['data'].reshape(s, s, s, s)
        self.net.forward()
        for blob in self.net.blobs.itervalues():
            for d in blob.data.shape:
                self.assertEqual(s, d)
