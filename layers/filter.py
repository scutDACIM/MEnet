# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:10:33 2016

@author: E81
"""

import numpy as np
import matplotlib.pyplot as plt
import sys,os
import cv2
import caffe

#im = caffe.io.load_image('/home/caisl/guided-filter-code-v1/img_smoothing/cat.bmp')
#plt.figure(1)
#plt.imshow(im)

class guided_filer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.r = params['radius']
        self.eps = params['epsilon']
        self.guided_type = params['guided_type']
#        self.batch_q = None
        
    def reshape(self, bottom, top):
        self.data = bottom[0].data
        self.cases, self.channels, self.height, self.width = self.data.shape
        top[0].reshape(self.cases, self.channels, self.height, self.width)
#        top[1].reshape(self.cases, self.channels, self.height, self.width)
#        top[2].reshape(self.cases, self.channels, self.height, self.width)
        
    def forward(self, bottom, top):
#        self.I = bottom[0].data
#        self.p = bottom[0].data
        if self.guided_type == 'gray':
            gf_01_data = self.guidedfilter(self.data, self.eps)
#            gf_001_data = self.guidedfilter(self.data - gf_01_data, 0.2)
            top[0].data[...] = self.data - gf_01_data
#            top[1].data[...] = gf_01_data[:, :, :, :]
#            top[2].data[...] = gf_001_data[:, :, :, :]
        
    def backward(self, top, propagate_down, bottom):
#        bottom[0].diff[...] = top[0].diff[...]
        pass
        
        
    def guidedfilter(self, data, eps):
        batch_q = np.zeros((self.cases, self.channels, self.height, self.width))
        r = self.r
#        eps = self.eps
        for i in range(self.cases):
            for j in range(self.channels):
                I = data[i, j, :, :] / 255.0
                p = data[i, j, :, :] / 255.0
                ones_array = np.ones([self.height, self.width])
                N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
                mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                cov_Ip = mean_Ip - mean_I * mean_p; # this is the covariance of (I, p) in each local patch.
                mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                var_I = mean_II - mean_I * mean_I
                a = cov_Ip / (var_I + eps) # Eqn. (5) in the paper;
                b = mean_p - a * mean_I; # Eqn. (6) in the paper;
                mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                q = mean_a * I + mean_b # Eqn. (8) in the paper;
                batch_q[i, j, :, :] = q * 255.0
        return batch_q

    def guidedfilter_color(self, I, p, r, eps):
        p = p[:,:,0]
        height, width = p.shape
        ones_array = np.ones([height, width])
        N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
       
        mean_I_r = cv2.boxFilter(I[:, :, 0], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
        mean_I_g = cv2.boxFilter(I[:, :, 1], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
        mean_I_b = cv2.boxFilter(I[:, :, 2], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
    
        mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
    
        mean_Ip_r = cv2.boxFilter(I[:, :, 0] * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
        mean_Ip_g = cv2.boxFilter(I[:, :, 1] * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
        mean_Ip_b = cv2.boxFilter(I[:, :, 2] * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
    
        # covariance of (I, p) in each local patch.
        cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
        cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
        cov_Ip_b = mean_Ip_b - mean_I_b * mean_p
    
        var_I_rr = cv2.boxFilter(I[:, :, 0] * I[:, :, 0], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N - mean_I_r *  mean_I_r
        var_I_rg = cv2.boxFilter(I[:, :, 0] * I[:, :, 1], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N - mean_I_r *  mean_I_g 
        var_I_rb = cv2.boxFilter(I[:, :, 0] * I[:, :, 2], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N - mean_I_r *  mean_I_b 
        var_I_gg = cv2.boxFilter(I[:, :, 1] * I[:, :, 1], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N - mean_I_g *  mean_I_g
        var_I_gb = cv2.boxFilter(I[:, :, 1] * I[:, :, 2], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N - mean_I_g *  mean_I_b
        var_I_bb = cv2.boxFilter(I[:, :, 2] * I[:, :, 2], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N - mean_I_b *  mean_I_b 
        #    return mean_I_r, mean_I_g, mean_I_b
    
        a = np.zeros([height, width, 3])
        for y in range(height):
            for x in range(width):
                Sigma = np.array([[var_I_rr[y, x], var_I_rg[y, x], var_I_rb[y, x]],
                                     [var_I_rg[y, x], var_I_gg[y, x], var_I_gb[y, x]],
                                    [var_I_rb[y, x], var_I_gb[y, x], var_I_bb[y, x]]])
                cov_Ip = np.array([[cov_Ip_r[y, x], cov_Ip_g[y, x], cov_Ip_b[y, x]]])
                a[y, x, :] = np.dot(cov_Ip, np.linalg.inv(Sigma + eps * np.diag(np.ones(3)))) # Eqn. (14) in the paper;
  
    
        b = mean_p - a[:, :, 0] * mean_I_r - a[:, :, 1] * mean_I_g - a[:, :, 2] * mean_I_b# Eqn. (15) in the paper;
        q = (cv2.boxFilter(a[:, :, 0], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) * I[:, :, 0] +
             cv2.boxFilter(a[:, :, 1], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) * I[:, :, 1] +
             cv2.boxFilter(a[:, :, 2], -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) * I[:, :, 2] +
             cv2.boxFilter(b, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)) / N
#       return q
#a = zeros(hei, wid, 3);
#for y=1:hei
#    for x=1:wid        
#        Sigma = [var_I_rr(y, x), var_I_rg(y, x), var_I_rb(y, x);
#            var_I_rg(y, x), var_I_gg(y, x), var_I_gb(y, x);
#            var_I_rb(y, x), var_I_gb(y, x), var_I_bb(y, x)];
#        %Sigma = Sigma + eps * eye(3);
#        
#        cov_Ip = [cov_Ip_r(y, x), cov_Ip_g(y, x), cov_Ip_b(y, x)];        
#        
#        a(y, x, :) = cov_Ip * inv(Sigma + eps * eye(3)); % Eqn. (14) in the paper;
#    end
#end
#


#I = caffe.io.load_image('/home/caisl/guided-filter-code-v1/img_feathering/toy.bmp')
#p = caffe.io.load_image('/home/caisl/guided-filter-code-v1/img_feathering/toy-mask.bmp')
#
#plt.figure(1)
#plt.imshow(I)
##p = I;
#I = np.array(I, dtype = np.float64)
#p = np.array(p, dtype = np.float64)
#r = 60;
#eps = 10**-6
#q = guidedfilter_color(I, p, r, eps)
#
#plt.figure(2)
#plt.imshow(p, cmap = plt.cm.gray)
#
#plt.figure(3)
#plt.imshow(q, cmap = plt.cm.gray)