import caffe
#import global_var as GV
import numpy as np


class layer_normalization(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        
        self.data = bottom[0].data
#        self.max = self.data.max((1,2,3))
        self.max = self.data.mean((1,2,3))
        self.min = self.data.min((1,2,3))
#        self.data = ((self.data.transpose(1,2,3,0) - self.min) / (self.max - self.min) * 2 - 1) * 128
#        self.data = self.data.transpose(3,0,1,2)
        self.data = self.data.transpose(1,2,3,0) - self.max
        self.data = self.data.transpose(3,0,1,2)
        top[0].reshape(*bottom[0].shape)
        top[1].reshape(*self.max.shape)
        top[2].reshape(*self.min.shape)
        
    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.max
        top[2].data[...] = self.min
        
    def backward(self, top, propagate_down, bottom):
        pass
