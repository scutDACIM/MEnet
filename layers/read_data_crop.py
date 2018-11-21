import caffe
import scipy.io as scio
import os.path as osp
import h5py
import numpy as np
import random
import read_binaryproto
#import read_lmdb
import matplotlib.pyplot as plt
import matplotlib.image as mping
from PIL import Image
import os
import global_var as GV
from scipy.misc import imresize
'''
this_dir = osp.dirname(__file__)
data_path = osp.join(this_dir,'data')
data_name = 'patches_1.mat'
data = [data_path, data_name]
data = h5py.File('/'.join(data))

index = data.keys()
labels = data[index[0]][0:2]
samples = data[index[1]][:]

yellolayer_dir = '/home/huangjb/mycaffe/data'
data_name = 'patches_1.mat'
data = [yellolayer_dir, data_name]
data = h5py.File('/'.join(data))
data_index = data.keys()
data[data_index[1]].shape
index_num = data_index.__len__()
data_mean = np.zeros((index_num,3,64,64))
data_mean[0] = np.sum(data[data_index[0]],axis=0)
data_mean[1] = np.sum(data[data_index[1]],axis=0)
print data_index[1]
print data[data_index[0]].len()
print data.keys().len
'''

class input_layer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.split = params['split']
        self.train_data_dir = params['train_data_dir']
        self.train_sobel_dir = params['train_sobel_dir']
        self.train_labels_dir = params['train_labels_dir']
        self.test_labels_dir = params['test_labels_dir']
        self.test_sobel_dir = params['test_sobel_dir']
        self.test_data_dir = params['test_data_dir']
        self.batch_size = params['batch_size']
        self.test_batch_size = params['test_batch_size']
        self.resize_size = params['resize_size']
        self.crop_ratio = 0.9
#        self.reshape_flag = params['reshape_flag']
#        self.reshape_size = params['reshape_size']
#        self.crop_size = params['crop_size']
#        self.train_batches = params['train_batches']
        if self.split == 'test':
            if os.path.exists(self.test_data_dir.split('/')[-2]):
                print 'The file ' + self.test_data_dir.split('/')[-2] + ' exists.' 
            else:
                os.mkdir(self.test_data_dir.split('/')[-2])
        GV.target_dir = self.test_data_dir.split('/')[-2]

        self.num_classes = 2
        self.train_timing = 0
        self.test_timing = 0
        self.train_images = os.listdir(osp.join(self.data_dir, self.train_data_dir))
#        self.train_labels = os.listdir(osp.join(self.data_dir, self.train_labels_dir))
        self.test_images = os.listdir(osp.join(self.data_dir, self.test_data_dir))
        
#        self.test_labels = os.listdir(osp.join(self.data_dir, self.test_labels_dir))
        self.train_images_num = self.train_images.__len__()
#        self.train_images_num = 1
        self.test_images_num = self.test_images.__len__()
        GV.test_images_num = self.test_images_num
        GV.normal_training = 1
        
    def reshape(self, bottom, top):
        if self.split == 'train':
#            self.train_timing = (self.train_timing + self.batch_size) % self.train_images_num 
            self.data = np.zeros([self.batch_size, 3, self.resize_size[0], self.resize_size[1]])
#            self.sobel_0 = np.zeros([self.batch_size, 1, self.resize_size[0], self.resize_size[1]])
#            self.sobel_1 = np.zeros([self.batch_size, 1, self.resize_size[0]/2, self.resize_size[1]/2])
#            self.sobel_2 = np.zeros([self.batch_size, 1, self.resize_size[0]/4, self.resize_size[1]/4])
#            self.sobel_3 = np.zeros([self.batch_size, 1, self.resize_size[0]/8, self.resize_size[1]/8])
#            self.sobel_4 = np.zeros([self.batch_size, 1, self.resize_size[0]/16, self.resize_size[1]/16])
#            self.sobel_5 = np.zeros([self.batch_size, 1, self.resize_size[0]/32, self.resize_size[1]/32])
            self.labels = np.zeros([self.batch_size, 1, self.resize_size[0], self.resize_size[1]])
            for i in range(self.batch_size):
                self.train_timing = (self.train_timing + 1) % self.train_images_num 
                orignial_image_data = mping.imread(osp.join(self.data_dir,  self.train_data_dir, self.train_images[self.train_timing]))     
#                orignial_image_sobel = mping.imread(osp.join(self.data_dir,  self.train_sobel_dir, self.train_images[self.train_timing]))
                orignial_image_labels = mping.imread(osp.join(self.data_dir, self.train_labels_dir, self.train_images[self.train_timing].split('.jpg')[0] + '.png'))
#                print osp.join(self.data_dir, self.train_labels_dir, self.train_images[self.train_timing].split('.jpg')[0] + '.png')                
                GV.data_name = self.train_images[self.train_timing].split('.jpg')[0]
                if len(orignial_image_labels.shape) == 3:
                    orignial_image_labels = orignial_image_labels.mean(axis = 2)
                
                if len(orignial_image_data.shape) != 3:
                    orignial_image_data = np.tile(orignial_image_data[:,:,np.newaxis], [1,1,3])
                if orignial_image_data.shape[:2] != orignial_image_labels.shape[:2]:
                    raise Exception('image and labels must be same size')
                height, width = orignial_image_data.shape[:2]
                
                height_add_flag = np.random.randint(0, 2)
                width_add_flag = np.random.randint(0, 2)
                if height_add_flag and width_add_flag:
                    height_add_flag = 0
                    width_add_flag = 0
                
                if height_add_flag or width_add_flag:
                    add_image = np.zeros([height * (1 + height_add_flag), width * (1 + width_add_flag), 3])
                    add_label = np.zeros([height * (1 + height_add_flag), width * (1 + width_add_flag)])
                    if height_add_flag and not width_add_flag:
                        add_image[:height,:width] = np.flipud(orignial_image_data)
                        add_image[height:2*height,:width] = orignial_image_data
                        add_label[:height,:width] = np.flipud(orignial_image_labels)
                        add_label[height:2*height,:width] = orignial_image_labels
                    elif not height_add_flag and width_add_flag:
                        add_image[:height,:width] = np.fliplr(orignial_image_data)
                        add_image[:height,width:2*width] = orignial_image_data
                        add_label[:height,:width] = np.fliplr(orignial_image_labels)
                        add_label[:height,width:2*width] = orignial_image_labels
                    elif height_add_flag and width_add_flag:
                        add_image[height:2*height,:width] = np.fliplr(orignial_image_data)
                        add_image[height:2*height,width:2*width] = orignial_image_data
                        add_image[:height,width:2*width] = np.flipud(orignial_image_data)
                        add_image[:height,:width] = np.fliplr(np.flipud(orignial_image_data))
    
                        add_label[height:2*height,:width] = np.fliplr(orignial_image_labels)
                        add_label[height:2*height,width:2*width] = orignial_image_labels
                        add_label[:height,width:2*width] = np.flipud(orignial_image_labels)
                        add_label[:height,:width] = np.fliplr(np.flipud(orignial_image_labels))
                    orignial_image_data = add_image
                    orignial_image_labels = add_label
                    height, width = orignial_image_data.shape[:2]
                    
                self.crop_ratio
                
                crop_height = random.randint(int(height * self.crop_ratio), height)
                crop_width = random.randint(int(width * self.crop_ratio), width)
                
                start_x = random.randint(0, height - crop_height)
                start_y = random.randint(0, width - crop_width)
                image_data = np.array(orignial_image_data[start_x : start_x + crop_height, start_y : start_y + crop_width])
#                image_sobel = orignial_image_sobel[end_x - int(height * tmp_crop_ratio) : end_x, end_y - int(width * tmp_crop_ratio) : end_y]
                image_labels = np.array(orignial_image_labels[start_x : start_x + crop_height, start_y : start_y + crop_width])
                flip = np.random.randint(0, 2)
                
                GV.a =image_labels   
                image_labels[np.where(image_labels>=0.1)] = 1
                image_labels[np.where(image_labels<0.1)] = 0
                GV.b = image_labels
                if flip == 1:
                    image_data = np.fliplr(image_data)
                    image_labels = np.fliplr(image_labels)
#                flip = np.random.randint(0, 2)
#                if flip == 1:
#                    image_data = np.flipud(image_data)
#                    image_labels = np.flipud(image_labels)
#                print image_data.shape
                    
                down_sample = 16
                desease_num = image_data.shape[0] - np.floor(image_data.shape[0] * 1. / down_sample) * down_sample
                desease_num = int(desease_num)
                if desease_num:
                    image_data = np.delete(image_data, range(desease_num), 0)
                    image_labels = np.delete(image_labels, range(desease_num), 0)
                desease_num = image_data.shape[1] - np.floor(image_data.shape[1] * 1. / down_sample) * down_sample
                desease_num = int(desease_num)
                if desease_num:
                    image_data = np.delete(image_data, range(desease_num), 1)
                    image_labels = np.delete(image_labels, range(desease_num), 1)
                    
#                print image_data.shape
#                dsa
                self.data = image_data.transpose(2, 0, 1)[np.newaxis]
#                self.sobel_0[i, 0] = image_sobel_0
#                self.sobel_1[i, 0] = image_sobel_1
#                self.sobel_2[i, 0] = image_sobel_2
#                self.sobel_3[i, 0] = image_sobel_3
#                self.sobel_4[i, 0] = image_sobel_4
#                self.sobel_5[i, 0] = image_sobel_5
                self.labels = image_labels[np.newaxis, np.newaxis]
#                das
            GV.image = self.data[-1].transpose(1,2,0)
            GV.labels = self.labels[-1,0]
#            plt.figure(2)
#            plt.subplot(221)
#            plt.imshow(GV.image/255)
#            plt.subplot(222)
#            plt.imshow(GV.labels)
#            plt.subplot(223)
#            plt.imshow(orignial_image_data/255.)
#            plt.subplot(224)
#            plt.imshow(orignial_image_labels)
            GV.zzz=orignial_image_labels
#            self.data = np.array(self.data, dtype = np.float32)
#            self.labels = np.array(self.labels, dtype = np.float32)
            

            
        elif self.split == 'test':
#            self.resize_size[0] = 64
#            self.resize_size[1] = 64
            self.data = np.zeros([self.test_batch_size, 3, self.resize_size[0], self.resize_size[1]])
#            self.sobel_0 = np.zeros([self.test_batch_size, 1, self.resize_size[0], self.resize_size[1]])
#            self.sobel_1 = np.zeros([self.test_batch_size, 1, self.resize_size[0]/2, self.resize_size[1]/2])
#            self.sobel_2 = np.zeros([self.test_batch_size, 1, self.resize_size[0]/4, self.resize_size[1]/4])
#            self.sobel_3 = np.zeros([self.test_batch_size, 1, self.resize_size[0]/8, self.resize_size[1]/8])
#            self.sobel_4 = np.zeros([self.test_batch_size, 1, self.resize_size[0]/16, self.resize_size[1]/16])
#            self.sobel_5 = np.zeros([self.test_batch_size, 1, self.resize_size[0]/32, self.resize_size[1]/32])
            self.labels = np.zeros([self.test_batch_size, 1, self.resize_size[0], self.resize_size[1]])
            for i in range(self.test_batch_size):
                self.test_timing = (self.test_timing + 1) % self.test_images_num
                suffix = osp.join(self.data_dir,  self.test_data_dir, self.test_images[self.test_timing]).split('.')[-1]
                if suffix == 'png':
                    image_data = mping.imread(osp.join(self.data_dir,  self.test_data_dir, self.test_images[self.test_timing])) * 255
                else:
                    image_data = mping.imread(osp.join(self.data_dir,  self.test_data_dir, self.test_images[self.test_timing]))
#                    image_sobel = mping.imread(osp.join(self.data_dir,  self.test_sobel_dir, self.test_images[self.test_timing]))
    #            print self.test_images[self.test_timing], suffix
                image_labels = mping.imread(osp.join(self.data_dir, self.test_labels_dir, self.test_images[self.test_timing].split('.' + suffix)[0] + '.png'))
                
                
                GV.data_name = self.test_images[self.test_timing].split('.')[0]
                GV.data_dir = osp.join(self.data_dir,  self.test_data_dir)
                print self.data_dir, self.test_labels_dir, self.test_images[self.test_timing], GV.data_name
                
                if len(image_data.shape) == 3:
                    if image_data.shape[2] != 3:
                        GV.image = image_data
                        image_data = image_data.mean()
                        image_data = np.tile(image_data[:,:,np.newaxis], [1,1,3])
                        
                elif len(image_data.shape) == 2:
                    image_data = np.tile(image_data[:,:,np.newaxis], [1,1,3])
                
                if len(image_labels.shape) == 3:
                    image_labels = image_labels[:,:,0]
                    
                down_sample = 16
                desease_num = image_data.shape[0] - np.floor(image_data.shape[0] * 1. / down_sample) * down_sample
                desease_num = int(desease_num)
                if desease_num:
                    image_data = np.delete(image_data, range(desease_num), 0)
                    image_labels = np.delete(image_labels, range(desease_num), 0)
                desease_num = image_data.shape[1] - np.floor(image_data.shape[1] * 1. / down_sample) * down_sample
                desease_num = int(desease_num)
                if desease_num:
                    image_data = np.delete(image_data, range(desease_num), 1)
                    image_labels = np.delete(image_labels, range(desease_num), 1)
                    
                GV.original = np.array(image_labels)
                image_labels[np.where(image_labels>=0.1)] = 1
                image_labels[np.where(image_labels<0.1)] = 0
                GV.image = image_data
                GV.labels = image_labels
                self.data = image_data.transpose(2, 0, 1)[np.newaxis]
                self.labels = image_labels[np.newaxis, np.newaxis]
                GV.xx = 'test'
                print 

        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.labels.shape)
        
        
    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.labels
        
    def backward(self, bottom, top):
        pass