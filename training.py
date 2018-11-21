import _init_paths
import caffe
import tools
import os.path as osp
import numpy as np
from caffe import layers as L, params as P, to_proto
import matplotlib.pyplot as plt
from PIL import Image
import global_var as GV
import matplotlib.image as mping
import h5py
import scipy.io as scio
import os
#das
def BN_scale_relu_conv(split, bottom, nout , ks=3, stride=1, pad=1, dilation = 1, conv_type = 'conv', in_place = False, lr = 2):
    if dilation != 1 and conv_type == 'conv':
        pad = ((dilation - 1) * (ks - 1) + ks - 1) / 2 
#        ks = (dilation - 1) * (ks - 1) + ks
#        dilation = 1
#        dsa
    if split == 'train':
        BN = L.BatchNorm(bottom, batch_norm_param = dict(use_global_stats = False),  in_place=in_place,
                         param = [dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)])
    else:
        BN = L.BatchNorm(bottom, batch_norm_param = dict(use_global_stats = True), in_place=in_place, 
                         param = [dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = True)
    relu = L.ReLU(scale, in_place=True) 
    
    if conv_type == 'conv':
        conv = L.Convolution(relu, kernel_size=ks, stride=stride, dilation = dilation, num_output=nout, pad=pad, bias_term = True, #, std = 0.000000001, mean = 0
                             weight_filler = dict(type='xavier'),
                             bias_filler = dict(type='constant'),
                             param=[dict(lr_mult=lr/2, decay_mult=1), dict( lr_mult=lr, decay_mult=0)])
    elif conv_type == 'dconv':
        conv = L.Deconvolution(relu, convolution_param=dict( weight_filler = dict(type='xavier'), dilation = dilation, num_output = nout, kernel_size = ks, stride = stride, pad = pad, bias_term=False), param=[dict(lr_mult=1)])

    return BN, scale, relu, conv

def conv_BN_scale_relu(split, bottom, nout , ks=3, stride=1, pad=1, dilation = 1, conv_type = 'conv', in_place = True, lr = 2):
    if dilation != 1 and conv_type == 'conv':
        pad = ((dilation - 1) * (ks - 1) + ks - 1) / 2 
#        ks = (dilation - 1) * (ks - 1) + ks
#        dilation = 1
    if conv_type == 'conv':
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride, dilation = dilation, num_output=nout, pad=pad, bias_term = True, #, std = 0.000000001, mean = 0
                             weight_filler = dict(type='xavier'),
                             bias_filler = dict(type='constant'),
                             param=[dict(lr_mult=lr/2, decay_mult=1), dict(lr_mult=lr, decay_mult=0)])
    elif conv_type == 'dconv':
        conv = L.Deconvolution(bottom, convolution_param=dict( weight_filler = dict(type='xavier'), dilation = dilation, num_output = nout, kernel_size = ks, stride = stride, pad = pad, bias_term=False), param=[dict(lr_mult=1)])
        
    if split == 'train':
        BN = L.BatchNorm(conv, batch_norm_param = dict(use_global_stats = False),  in_place=in_place,
                         param = [dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)])
    else:
        BN = L.BatchNorm(conv, batch_norm_param = dict(use_global_stats = True), in_place=in_place, 
                         param = [dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = in_place)
    relu = L.ReLU(scale, in_place=in_place)
    return conv, BN, scale, relu

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

test_num = 54
GV.test_data_dir = '0305_data_split/Images_test'  #'0322data/Train-n/Images_train' 0305_data_split  0331data/Train-n/Images_train

GV.test_labels_dir = '0305_data_split/Images_test'


GV.lr = 0.1
def simpleNN(split):
    nout = 32
    repeat = 2
    dilation = 1
    ks = 3
    pad = 1
    block_nums = 1
    block_nums = block_nums - 1
    
    GV.repeat = repeat
    data, labels = L.Python(module = 'read_data1', 
                                   layer = 'input_layer',
                                   ntop = 2,
                                   param_str = str(dict(split=split, 
                                                        #data_dir = './data/../../',
                                                        data_dir = '/workspace/final_4_DML_model',
                                                        train_data_dir = '0408data/Images_train',
                                                        train_sobel_dir = '0408data/Images_train',
                                                        train_labels_dir = '0408data/GT_train',
                                                        test_data_dir = GV.test_data_dir,#
                                                        test_sobel_dir = '0408data/Images_train_SOBEL',
                                                        test_labels_dir = GV.test_labels_dir,#
                                                        batch_size = 5,
                                                        test_batch_size = 1,
                                                        resize_size = [224, 224]
                                                        )))
#    gf0 = L.Python(data, module = 'filter', 
#                                   layer = 'guided_filer',
#                                   ntop = 1,
#                                   param_str = str(dict(split=split, 
#                                                        radius = 15,
#                                                        epsilon = 0.3,
#                                                        guided_type = 'gray'
#                                                        )))
    set_dilation = 1
    dilation = set_dilation
#    gf0 = L.Concat(gf0, sobel_0)
    result = data
    a, b, c, result = conv_BN_scale_relu(split, bottom = result, nout = nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
    
#    data = L.Concat(data, sobel_0)
    a, b, c, data_f = BN_scale_relu_conv(split, bottom = data, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, mask_f = BN_scale_relu_conv(split, bottom = data, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    data_f = L.Eltwise(data_f, mask_f, operation = P.Eltwise.PROD)
    for i in range(repeat): #224
        a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)  
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)        
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
        
    scale0 = result
#    scale0 = L.Concat(scale0, sobel_0)
    a, b, c, scale0_m = BN_scale_relu_conv(split, bottom = scale0, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale0_d_mask = BN_scale_relu_conv(split, bottom = scale0, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale0_m = L.Eltwise(scale0_m, scale0_d_mask, operation = P.Eltwise.PROD)
    for i in range(repeat):#112
        if i == 0:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 2 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)  
            a, b, c, result = BN_scale_relu_conv(split, bottom = result, nout = 2 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
        else:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)        
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
    scale1 = result
#    scale1 = L.Concat(scale1, sobel_1)
    a, b, c, scale1_m = BN_scale_relu_conv(split, bottom = scale1, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale1_d_mask = BN_scale_relu_conv(split, bottom = scale1, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale1_m = L.Eltwise(scale1_m, scale1_d_mask, operation = P.Eltwise.PROD)
    for i in range(repeat):#56
        if i == 0:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 4 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
            a, b, c, result = BN_scale_relu_conv(split, bottom = result, nout = 4 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
        else:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
    scale2 = result
#    scale2 = L.Concat(scale2, sobel_2)
    a, b, c, scale2_m = BN_scale_relu_conv(split, bottom = scale2, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale2_d_mask = BN_scale_relu_conv(split, bottom = scale2, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale2_m = L.Eltwise(scale2_m, scale2_d_mask, operation = P.Eltwise.PROD)
    for i in range(repeat):#28
        if i == 0:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 8 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
            a, b, c, result = BN_scale_relu_conv(split, bottom = result, nout = 8 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
        else:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
    scale3 = result
#    scale3 = L.Concat(scale3, sobel_3)
    a, b, c, scale3_m = BN_scale_relu_conv(split, bottom = scale3, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale3_d_mask = BN_scale_relu_conv(split, bottom = scale3, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale3_m = L.Eltwise(scale3_m, scale3_d_mask, operation = P.Eltwise.PROD)
    for i in range(repeat):#14
        if i == 0:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 16 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
            a, b, c, result = BN_scale_relu_conv(split, bottom = result, nout = 16 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
        else:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
    scale4 = result
#    scale4 = L.Concat(scale4, sobel_4)
    a, b, c, scale4_m = BN_scale_relu_conv(split, bottom = scale4, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale4_d_mask = BN_scale_relu_conv(split, bottom = scale4, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale4_m = L.Eltwise(scale4_m, scale4_d_mask, operation = P.Eltwise.PROD)
    for i in range(repeat):#7
        if i == 0:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 32 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
            a, b, c, result = BN_scale_relu_conv(split, bottom = result, nout = 32 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
        else:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 32 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 32 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
    scale5 = result
#    scale5 = L.Concat(scale5, sobel_5)
    a, b, c, scale5_m = BN_scale_relu_conv(split, bottom = scale5, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale5_d_mask = BN_scale_relu_conv(split, bottom = scale5, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale5_m = L.Eltwise(scale5_m, scale5_d_mask, operation = P.Eltwise.PROD)
    dilation = 1
    a, b, c, result = BN_scale_relu_conv(split, bottom = result, nout = 32 * nout, ks=7, stride=1, pad = 0, dilation = dilation, in_place = False)
    
    a, b, c, result = BN_scale_relu_conv(split, bottom = result, nout = 32 * nout, ks =7, stride=1, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)

    a, b, c, scale5_u = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale5_u_mask = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale5_u = L.Eltwise(scale5_u, scale5_u_mask, operation = P.Eltwise.PROD)
    result = L.Concat(result, scale5)
    for i in range(repeat):#14
        if i == 0:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 16 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
            a, b, c, result  = BN_scale_relu_conv(split, bottom = result, nout = 16 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
        else:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation,in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation ,in_place = True)
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
    a, b, c, scale4_u = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale4_u_mask = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale4_u = L.Eltwise(scale4_u, scale4_u_mask, operation = P.Eltwise.PROD)
    result = L.Concat(result, scale4)
    for i in range(repeat):#28
        if i == 0:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 8 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
            a, b, c, result  = BN_scale_relu_conv(split, bottom = result, nout = 8 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
        else:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation,in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
    a, b, c, scale3_u = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale3_u_mask = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale3_u = L.Eltwise(scale3_u, scale3_u_mask, operation = P.Eltwise.PROD)
    result = L.Concat(result, scale3)
    for i in range(repeat):#56
        if i == 0:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 4 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
            a, b, c, result  = BN_scale_relu_conv(split, bottom = result, nout = 4 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
        else:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
    a, b, c, scale2_u = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale2_u_mask = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale2_u = L.Eltwise(scale2_u, scale2_u_mask, operation = P.Eltwise.PROD)
    result = L.Concat(result, scale2)
    for i in range(repeat):#112
        if i == 0:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 2 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
            a, b, c, result  = BN_scale_relu_conv(split, bottom = result, nout = 2 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
        else:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
    a, b, c, scale1_u = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale1_u_mask = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale1_u = L.Eltwise(scale1_u, scale1_u_mask, operation = P.Eltwise.PROD)
    result = L.Concat(result, scale1)
    for i in range(repeat):#224
        if i == 0:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 1 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
            a, b, c, result  = BN_scale_relu_conv(split, bottom = result, nout = 1 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
        else:
            a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 1 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 1 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
        result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
    a, b, c, scale0_u = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    a, b, c, scale0_u_mask = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
#    scale0_u = L.Eltwise(scale0_u, scale0_u_mask, operation = P.Eltwise.PROD)
#    result = L.Concat(result, scale0)
    
#    a, b, c, result = BN_scale_relu_conv(split, bottom = result, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
    result = L.Python(data_f, scale0_m, scale1_m, scale2_m, scale3_m, scale4_m, scale5_m, scale0_u, scale1_u, scale2_u, scale3_u, scale4_u, scale5_u,
                                   ntop = 1,
                                   module = 'resize', 
                                   layer = 'resize_to_same_size',
                                   param_str = str(dict()))
#    metric = result                           
    a, b, metric, c = conv_BN_scale_relu(split, bottom = result, nout = 16, ks=3, stride=1, pad = 1, dilation = dilation, in_place = True)
    metric = L.TanH(metric, in_place= True)
    
    
    a, b, sal, d = conv_BN_scale_relu(split, bottom = result, nout = 2, ks=3, stride=1, pad = 1, dilation = dilation, in_place = True)
    result = L.Python(metric, sal, labels,
                                   ntop = 1,
                                   module = 'pyloss', 
                                   layer = 'EuclideanLossLayer',
                                   param_str = str(dict(crop_size = [64, 64],
                                                        batches = 20,
                                                        sup_threshold = 3,
                                                        inf_threshold = 0
                                                       )),loss_weight=1)
    return to_proto(result)

def store_curve_value(snapshot_name, train_dir, test_dir, data_iter = 10,  iter_step = 500):
    loss_file = snapshot_name + '.h5'
    start_index = 0
    if os.path.exists(loss_file):
        print 'The file ' + loss_file  + '.h5 exists.' 
        h5file = h5py.File(loss_file, 'r')
        train_metric_loss = h5file['train_metric_loss'][...]
        train_sal_loss = h5file['train_sal_loss'][...]
        test_metric_loss = h5file['test_metric_loss'][...]
        test_sal_loss = h5file['test_sal_loss'][...]
        h5file.close()
        start_index = (train_metric_loss.shape[0]) * iter_step
        train_metric_loss = list(train_metric_loss)
        train_sal_loss = list(train_sal_loss)
        test_metric_loss = list(test_metric_loss)
        test_sal_loss = list(test_sal_loss)
        print start_index
    else:
        train_metric_loss = []
        train_sal_loss = []
        test_metric_loss = []
        test_sal_loss = []
        print 'The file ' + loss_file + ' has set up.'
    for index in range(start_index + iter_step, 110000 + iter_step, iter_step):
#        print index
        weight = osp.join('model', 'snapshot', snapshot_name + '_iter_' + str(index) + '.caffemodel')
        train_net = caffe.Net(str(train_dir), str(weight), caffe.TEST)
        test_net = caffe.Net(str(test_dir), str(weight), caffe.TEST)
        tmp_train_metric_loss = 0
        tmp_train_sal_loss = 0
        tmp_test_metric_loss = 0
        tmp_test_sal_loss = 0
        for forward_iter in range(data_iter):
            train_net.forward()
            test_net.forward()
            tmp_train_metric_loss = tmp_train_metric_loss + train_net.blobs['Python4'].data[0]
            tmp_train_sal_loss = tmp_train_sal_loss + train_net.blobs['Python4'].data[1]
            tmp_test_metric_loss = tmp_test_metric_loss + test_net.blobs['Python4'].data[0]
            tmp_test_sal_loss = tmp_test_sal_loss + test_net.blobs['Python4'].data[1]
        train_metric_loss.append(train_metric_loss / data_iter)
        train_sal_loss.append(train_sal_loss / data_iter)
        test_metric_loss.append(test_metric_loss / data_iter)
        test_sal_loss.append(test_sal_loss / data_iter)
#        if index % 20 == 0:
        print 'saving loss data', train_metric_loss, train_sal_loss, test_metric_loss, test_sal_loss
#        da
        h5file = h5py.File(loss_file, 'w')
        h5file.create_dataset('train_metric_loss', data = np.array(train_metric_loss), dtype = np.float32)
        h5file.create_dataset('train_sal_loss', data = np.array(train_sal_loss), dtype = np.float32)
        h5file.create_dataset('test_metric_loss', data = np.array(test_metric_loss), dtype = np.float32)
        h5file.create_dataset('test_sal_loss', data = np.array(test_sal_loss), dtype = np.float32)
        h5file.close()
#    store_curve_value(snapshot_name, train_dir, test_dir)
#    das
def store_curve_to_mat(snapshot_name, train_dir, test_dir, data_iter = 100,  iter_step = 10000):
    snapshot_name = snapshot_name.split('_iter_')[0]
    loss_file = snapshot_name + '.mat'
    start_index = 0
    if os.path.exists(loss_file):
        print 'The file ' + loss_file  + '.mat exists.' 
        matfile = scio.loadmat(loss_file)
        train_metric_loss = matfile['train_metric_loss']
        train_sal_loss = matfile['train_sal_loss']
        test_metric_loss = matfile['test_metric_loss']
        test_sal_loss = matfile['test_sal_loss']
        start_index = (train_metric_loss.shape[0]) * iter_step
        train_metric_loss = list(train_metric_loss)
        train_sal_loss = list(train_sal_loss)
        test_metric_loss = list(test_metric_loss)
        test_sal_loss = list(test_sal_loss)
        print start_index
    else:
        train_metric_loss = []
        train_sal_loss = []
        test_metric_loss = []
        test_sal_loss = []
        print 'The file ' + loss_file + ' has set up.'
    for index in range(start_index + iter_step, 110000 + iter_step, iter_step):
#        print index
        weight = osp.join('model', 'snapshot', snapshot_name + '_iter_' + str(index) + '.caffemodel')
        train_net = caffe.Net(str(train_dir), str(weight), caffe.TEST)
        test_net = caffe.Net(str(test_dir), str(weight), caffe.TEST)
        tmp_train_metric_loss = 0
        tmp_train_sal_loss = 0
        tmp_test_metric_loss = 0
        tmp_test_sal_loss = 0
        for forward_iter in range(data_iter):
            train_net.forward()
            test_net.forward()
            tmp_train_metric_loss = tmp_train_metric_loss + train_net.blobs['Python4'].data[0]
            tmp_train_sal_loss = tmp_train_sal_loss + train_net.blobs['Python4'].data[1]
            tmp_test_metric_loss = tmp_test_metric_loss + test_net.blobs['Python4'].data[0]
            tmp_test_sal_loss = tmp_test_sal_loss + test_net.blobs['Python4'].data[1]
        train_metric_loss.append(tmp_train_metric_loss / data_iter)
        train_sal_loss.append(tmp_train_sal_loss / data_iter)
        test_metric_loss.append(tmp_test_metric_loss / data_iter)
        test_sal_loss.append(tmp_test_sal_loss / data_iter)
#        if index % 20 == 0:
        print 'saving loss data', train_metric_loss, train_sal_loss, test_metric_loss, test_sal_loss
#        da
        scio.savemat(loss_file, {'train_metric_loss': train_metric_loss,
                                   'train_sal_loss': train_sal_loss,
                                   'test_metric_loss': test_metric_loss,
                                   'test_sal_loss': test_sal_loss})
#        h5file = h5py.File(loss_file, 'w')
#        h5file.create_dataset('train_metric_loss', data = np.array(train_metric_loss), dtype = np.float32)
#        h5file.create_dataset('train_sal_loss', data = np.array(train_sal_loss), dtype = np.float32)
#        h5file.create_dataset('test_metric_loss', data = np.array(test_metric_loss), dtype = np.float32)
#        h5file.create_dataset('test_sal_loss', data = np.array(test_sal_loss), dtype = np.float32)
#        h5file.close()
    
def make_net(snapshot_name = None):
#    this_dir + '/model/train.prototxt'
    if not snapshot_name:
        train_prototxt_dir = './model/train.prototxt'
        test_prototxt_dir = './model/test.prototxt'
    else:
        train_prototxt_dir = './model/train_{}.prototxt'.format(snapshot_name)
        test_prototxt_dir = './model/test_{}.prototxt'.format(snapshot_name)
    with open(train_prototxt_dir, 'w') as f:
        f.write(str(simpleNN('train')))
    with open(test_prototxt_dir, 'w') as f:
        f.write(str(simpleNN('test')))
    solver_dir = './model/solver.prototxt'
    return train_prototxt_dir, test_prototxt_dir, solver_dir
    
GV.hard_samples = []
this_dir = osp.dirname(__file__)
GV.test_nums = 0
GV.test_images_num = 2
GV.abnormal_files = []

watch_single_result = True

GV.phase = 'train'

#GV.phase = 'test'

GV.test_dir = '/workspace/final_4_DML_model/0305_data_split/'
GV.target_data_dir = '.'
if __name__ == '__main__':
    make_net()

    train_dir, test_dir, solver_dir = make_net()
    caffe.set_device(0)
    caffe.set_mode_gpu()

    
    #snapshot_name = 'f4_lr_0.1_0408_pyloss_data_sal_split_first5times_pyloss_plus_num_finalout16_two_Loss_ks3_HardNeg_crop_nout32_repeat2_batch5_dilation1_1e-8_iter_110000'
    snapshot_name = 'attention_iter_110000'


    weight = '/workspace/final_4_DML_model/model/snapshot/' + str(snapshot_name) + '.caffemodel'

    state = '/workspace/final_4_DML_model/model/snapshot/' + str(snapshot_name) + '.solverstate'

    if GV.phase == 'test':
      net = caffe.Net(str(test_dir), str(weight), caffe.TEST)
      net.forward()

    elif GV.phase == 'train':
      solver = caffe.SGDSolver(str(solver_dir))
      solver.step(1)
      for i in range(1100):
         solver.step(100)
    print 'done!'


'''''''''
    test_files = os.listdir(osp.join(GV.test_dir))    
    GV.target_data_dir = snapshot_name
    if not os.path.exists(os.path.join(GV.target_data_dir)):
        os.mkdir(os.path.join(GV.target_data_dir))
    for i in range(len(test_files)):
        test_database = os.listdir(osp.join(GV.test_dir, test_files[i]))
        if not os.path.exists(os.path.join(GV.target_data_dir, test_files[i])):
            os.mkdir(os.path.join(GV.target_data_dir, test_files[i]))            
#        sort_key = ['D', 'H', 'E', 'M', 'P', 'S']
#        sort_key = ['D']
#        test_database = sorted(test_database, key= lambda x:sort_key.index(x[0]))
        for j in range(len(test_database)):
            GV.test_data_dir = osp.join(GV.test_dir, test_files[i], test_database[j] + '/')
            GV.test_labels_dir = osp.join(GV.test_dir, test_files[i], test_database[j] + '/')
#            GV.test_labels_dir = osp.join(GV.test_label_dir, test_files[i], test_database[j] + '/')
            if not os.path.exists(os.path.join(GV.target_data_dir, test_files[i], test_database[j])):
                os.mkdir(os.path.join(GV.target_data_dir, test_files[i], test_database[j]))            
            GV.target_dir = osp.join(GV.target_data_dir, test_files[i], test_database[j] + '/')
            train_dir, test_dir, solver_dir = make_net(snapshot_name = snapshot_name)
            net = caffe.Net(str(test_dir), str(weight), caffe.TEST)
#            dsd
            net.forward()
            for p in range(GV.test_images_num):
                net.forward()

'''''''''



    
    
    
    
