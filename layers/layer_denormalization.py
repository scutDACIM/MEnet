import caffe
#import global_var as GV
import numpy as np


class layer_denormalization(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        
        self.data = bottom[0].data
        self.max = bottom[1].data
        self.min = bottom[2].data
#        self.data = ((self.data.transpose(1,2,3,0) - self.min) / (self.max - self.min) * 2 - 1) * 128
#        self.data = (self.data.transpose(1,2,3,0) / 128 + 1) / 2 * (self.max - self.min) + self.min
        self.data = self.data.transpose(1,2,3,0) + self.max
        self.data = self.data.transpose(3,0,1,2) 
        top[0].reshape(*bottom[0].shape)
        
    def forward(self, bottom, top):
        top[0].data[...] = self.data
        
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff[...]
#        bottom[0].diff[...] = top[0].diff[...]
#        tmp = top[0].diff[...]
#        tmp = tmp.transpose(1,2,3,0) * (self.max - self.min) / 2 / 128
#        bottom[0].diff[...] = tmp.transpose(3,0,1,2)
#        bottom[1].diff[...] = np.zeros_like(self.max)
#        bottom[2].diff[...] = np.zeros_like(self.min)
