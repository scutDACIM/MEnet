import caffe
#import global_var as GV
import numpy as np
import global_var as GV


class select_region(caffe.Layer):
        
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.crop_size = params['crop_size']
        self.num_classes = 2
        

    def reshape(self, bottom, top):
        self.num_cases, self.channels, self.height, self.width = bottom[0].data.shape
        self.data = bottom[0].data
        self.labels = bottom[1].data
        
        self.select_matrix = np.zeros([1, self.channels, self.height, self.width])
        self.output = np.zeros([1, 2 * self.channels, self.crop_size[0], self.crop_size[1]])
        self.output_labels = np.zeros([1, self.num_classes, self.crop_size[0], self.crop_size[1]])
        
        self.crop_x = np.random.randint(0, self.height - self.crop_size[0], 2)
        self.crop_y = np.random.randint(0, self.width - self.crop_size[1], 2)
        
        tmp_output_labels = self.labels[:, :, self.crop_x[0] : self.crop_x[0] + self.crop_size[0], self.crop_y[0] : self.crop_y[0] + self.crop_size[1]] - \
                                self.labels[:, :, self.crop_x[1] : self.crop_x[1] + self.crop_size[0], self.crop_y[1] : self.crop_y[1] + self.crop_size[1]]
#        for i in range(self.num_classes):
        self.output_labels[:, 0] = (tmp_output_labels[0] != 0) * 1.0
        self.output_labels[:, 1] = 1 - self.output_labels[:, 0]
            
#            GV.d = tmp_output_labels
#        GV.c = self.output_labels
            
        for i in range(2):
            self.output[0, i * self.channels : i * self.channels + self.channels] = \
                self.data[0, :, self.crop_x[i] : self.crop_x[i] + self.crop_size[0], self.crop_y[i] : self.crop_y[i] + self.crop_size[1]]
#            self.select_matrix[i, :, self.crop_x[i] : self.crop_x[i] + self.crop_size[0], self.crop_y[i] : self.crop_y[i] + self.crop_size[1]] = 1
#        print self.output.shape, self.output_labels.shape
#        das
        top[0].reshape(*self.output.shape)
        top[1].reshape(*self.output_labels.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.output
        top[1].data[...] = self.output_labels
        
    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            self.select_matrix[0, :, self.crop_x[i] : self.crop_x[i] + self.crop_size[0], self.crop_y[i] : self.crop_y[i] + self.crop_size[1]] += \
                top[0].diff[0, i * self.channels : i * self.channels + self.channels]
        bottom[0].diff[...] = self.select_matrix
        
        
        
        
        
        
        