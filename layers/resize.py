import caffe
import numpy as np
import global_var as GV
from scipy.misc import imresize
import random
import matplotlib.pyplot as plt
from PIL import Image


class resize_to_same_size(caffe.Layer):
        
    def setup(self, bottom, top):
        pass
    
    def simple_resize(self, data, target_size):
        original_size = data.shape
        resize_data = np.zeros([original_size[0], original_size[1], target_size[2], target_size[3]])
        x_step = target_size[2] / original_size[2]
        y_step = target_size[3] / original_size[3]
        for i in range(original_size[2]):
            for j in range(original_size[3]):
                resize_data[:, :, i * x_step : (i + 1)* x_step, j * y_step : (j + 1) * y_step] = data[:, :, i, j, np.newaxis, np.newaxis]
        return resize_data

    def calcu_diff(self, total_diff, ID, size_record):
        x_step = size_record[0][2] / size_record[ID][2]
        y_step = size_record[0][3] / size_record[ID][3]
        channel_start = 0
        for i in range(ID):
            channel_start += size_record[i][1]
            
        if size_record[ID] == size_record[0]:
                diff = total_diff[:, channel_start : channel_start + size_record[ID][1]]
        else:
            diff = np.zeros(size_record[ID])
            for i in range(size_record[ID][2]):
                for j in range(size_record[ID][3]):
                    diff[:, :, i, j] = np.sum(total_diff[:, channel_start : channel_start + size_record[ID][1], i * x_step : (i + 1) * x_step, j * y_step : (j + 1) * y_step].reshape(size_record[ID][0], size_record[ID][1], x_step * y_step), axis = 2)

        return diff
    
    def reshape(self, bottom, top):
        self.N = len(bottom)
        tagert_size = bottom[0].data.shape
        self.data = bottom[0].data
        self.size_record = [tagert_size]
        for i in range(1, self.N):
            if bottom[i].data.shape != tagert_size:
                resized_data = self.simple_resize(bottom[i].data, tagert_size)
            else:
                resized_data = bottom[i].data
            self.size_record.append(bottom[i].data.shape)
            self.data = np.concatenate((self.data, resized_data), axis = 1)
        top[0].reshape(*self.data.shape)
        
    def forward(self, bottom, top):
        top[0].data[...] = self.data
        
    def backward(self, top, propagate_down, bottom):
        GV.record_diff = []
        for i in range(self.N):
            bottom[i].diff[...] = self.calcu_diff(top[0].diff, i, self.size_record)
            GV.record_diff.append(self.calcu_diff(top[0].diff, i, self.size_record))
            GV.all_diff = top[0].diff
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        