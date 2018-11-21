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
        self.resize_ratio = 2
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
                
                if orignial_image_data.shape[:2] != orignial_image_labels.shape[:2]:
                    raise Exception('image and labels must be same size')
                height, width = orignial_image_data.shape[:2]
                
                tmp_crop_ratio = random.randint(int(100 * self.crop_ratio), 100) / 100.
                end_x = random.randint(int(height * tmp_crop_ratio), height)
                end_y = random.randint(int(width * tmp_crop_ratio), width)
                image_data = orignial_image_data[end_x - int(height * tmp_crop_ratio) : end_x, end_y - int(width * tmp_crop_ratio) : end_y]
#                image_sobel = orignial_image_sobel[end_x - int(height * tmp_crop_ratio) : end_x, end_y - int(width * tmp_crop_ratio) : end_y]
                image_labels = orignial_image_labels[end_x - int(height * tmp_crop_ratio) : end_x, end_y - int(width * tmp_crop_ratio) : end_y]
                
#                end_x = random.randint(int(height * self.crop_ratio), height)
#                end_y = random.randint(int(width * self.crop_ratio), width)
#                image_data = orignial_image_data[:end_x, :end_y]
#                image_sobel = orignial_image_sobel[:end_x, :end_y]
#                image_labels = orignial_image_labels[:end_x, :end_y]
                
#                plt.figure(1)
#                plt.subplot(221)
#                plt.imshow(orignial_image_data)
#                plt.subplot(222)
#                plt.imshow(orignial_image_labels)
#                plt.subplot(223)
#                plt.imshow(image_sobel)
#                plt.subplot(224)
#                plt.imshow(image_labels)
#                GV.data_name = self.test_images[self.train_timing].split('.')[0]
#                print self.data_dir,  self.train_data_dir, self.train_images[self.train_timing]
#                image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1], 3])
#                image_labels = imresize(np.array(image_labels, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                flip = np.random.randint(0, 2)
                if len(image_data.shape) == 3:
                    image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1], 3])
#                    image_sobel_0 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
#                    image_sobel_1 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/2, self.resize_size[1]/2])
#                    image_sobel_2 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/4, self.resize_size[1]/4])
#                    image_sobel_3 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/8, self.resize_size[1]/8])
#                    image_sobel_4 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/16, self.resize_size[1]/16])
#                    image_sobel_5 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/32, self.resize_size[1]/32])
                else:
                    image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                    image_data = np.tile(image_data[:,:,np.newaxis], [1,1,3])
#                    image_sobel_0 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
#                    image_sobel_1 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/2, self.resize_size[1]/2])
#                    image_sobel_2 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/4, self.resize_size[1]/4])
#                    image_sobel_3 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/8, self.resize_size[1]/8])
#                    image_sobel_4 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/16, self.resize_size[1]/16])
#                    image_sobel_5 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/32, self.resize_size[1]/32])
                    GV.abnormal_files.append(GV.data_name)
#                image_data = np.concatenate((image_data, image_sobel[:,:,np.newaxis]), axis = 2)
                if len(image_labels.shape) == 3:
                    image_labels = imresize(np.array(image_labels[:,:,0], dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                    GV.abnormal_files.append(GV.data_name)
                else:
                    image_labels = imresize(np.array(image_labels, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                image_labels[np.where(image_labels>0)] = 1
                if flip == 1:
                    image_data = np.fliplr(image_data)
                    image_labels = np.fliplr(image_labels)
#                flip = np.random.randint(0, 2)
#                if flip == 1:
#                    image_data = np.flipud(image_data)
#                    image_labels = np.flipud(image_labels)
                self.data[i] = image_data.transpose(2, 0, 1)
#                self.sobel_0[i, 0] = image_sobel_0
#                self.sobel_1[i, 0] = image_sobel_1
#                self.sobel_2[i, 0] = image_sobel_2
#                self.sobel_3[i, 0] = image_sobel_3
#                self.sobel_4[i, 0] = image_sobel_4
#                self.sobel_5[i, 0] = image_sobel_5
                self.labels[i, 0] = image_labels
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
#                print i
                self.test_timing = (self.test_timing + 1) % self.test_images_num
                suffix = osp.join(self.data_dir,  self.test_data_dir, self.test_images[self.test_timing]).split('.')[-1]
                if suffix == 'png':
                    image_data = mping.imread(osp.join(self.data_dir,  self.test_data_dir, self.test_images[self.test_timing])) * 255
#                    image_sobel = mping.imread(osp.join(self.data_dir,  self.test_sobel_dir, self.test_images[self.test_timing]))
                else:
                    image_data = mping.imread(osp.join(self.data_dir,  self.test_data_dir, self.test_images[self.test_timing]))
#                    image_sobel = mping.imread(osp.join(self.data_dir,  self.test_sobel_dir, self.test_images[self.test_timing]))
    #            print self.test_images[self.test_timing], suffix
                image_labels = mping.imread(osp.join(self.data_dir, self.test_labels_dir, self.test_images[self.test_timing].split('.' + suffix)[0] + '.png'))
    #            print osp.join(self.data_dir, self.test_labels_dir, self.test_images[self.test_timing].split('.' + suffix)[0] + '.png')
                
                image_labels[np.where(image_labels>0.1)] = 1
                
#                hegiht, width, _ = image_labels.shape
#                max_height = 64
#                if height > max_height:
#                    image_labels = imresize(image_labels, [max_height, max_height * width / height, 3])
#                    image_labels = imresize(image_labels, [max_height, max_height * width / height, 3])
                
                GV.data_name = self.test_images[self.test_timing].split('.')[0]
                GV.data_dir = osp.join(self.data_dir,  self.test_data_dir)
                print self.data_dir, self.test_labels_dir, self.test_images[self.test_timing], GV.data_name
                GV.a = self.data_dir
                GV.b = self.test_labels_dir
                GV.c = self.test_images[self.test_timing]
                GV.d = GV.data_name
    #            print 'hello', GV.data_name
                if len(image_data.shape) == 3:
                    if image_data.shape[2] != 3:
                        GV.image = image_data
                        image_data = imresize(np.array(image_data[:,:,0] * 255, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                        image_data = np.tile(image_data[:,:,np.newaxis], [1,1,3])
#                        image_sobel_0 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
#                        image_sobel_1 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/2, self.resize_size[1]/2])
#                        image_sobel_2 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/4, self.resize_size[1]/4])
#                        image_sobel_3 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/8, self.resize_size[1]/8])
#                        image_sobel_4 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/16, self.resize_size[1]/16])
#                        image_sobel_5 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/32, self.resize_size[1]/32])
                    else:
                        image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1], 3])
#                        image_sobel_0 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
#                        image_sobel_1 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/2, self.resize_size[1]/2])
#                        image_sobel_2 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/4, self.resize_size[1]/4])
#                        image_sobel_3 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/8, self.resize_size[1]/8])
#                        image_sobel_4 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/16, self.resize_size[1]/16])
#                        image_sobel_5 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/32, self.resize_size[1]/32])
    #                    dasda
                elif len(image_data.shape) == 2:
                    image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                    image_data = np.tile(image_data[:,:,np.newaxis], [1,1,3])
#                    image_sobel_0 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
#                    image_sobel_1 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/2, self.resize_size[1]/2])
#                    image_sobel_2 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/4, self.resize_size[1]/4])
#                    image_sobel_3 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/8, self.resize_size[1]/8])
#                    image_sobel_4 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/16, self.resize_size[1]/16])
#                    image_sobel_5 = imresize(np.array(image_sobel, dtype = np.uint8), [self.resize_size[0]/32, self.resize_size[1]/32])
                
                if len(image_labels.shape) == 3:
                    image_labels = imresize(np.array(image_labels[:,:,0], dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                else:
                    image_labels = imresize(np.array(image_labels, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
#                image_data = np.concatenate((image_data, image_sobel[:,:,np.newaxis]), axis = 2)
                GV.image = image_data
                self.data[i] = image_data.transpose(2, 0, 1)
#                self.sobel_0[i, 0] = image_sobel_0
#                self.sobel_1[i, 0] = image_sobel_1
#                self.sobel_2[i, 0] = image_sobel_2
#                self.sobel_3[i, 0] = image_sobel_3
#                self.sobel_4[i, 0] = image_sobel_4
#                self.sobel_5[i, 0] = image_sobel_5
                self.labels[i, 0] = image_labels

        top[0].reshape(*self.data.shape)
#        top[1].reshape(*self.sobel_0.shape)
#        top[2].reshape(*self.sobel_1.shape)
#        top[3].reshape(*self.sobel_2.shape)
#        top[4].reshape(*self.sobel_3.shape)
#        top[5].reshape(*self.sobel_4.shape)
#        top[6].reshape(*self.sobel_5.shape)
        top[1].reshape(*self.labels.shape)
        
        
    def forward(self, bottom, top):
        top[0].data[...] = self.data
#        top[1].data[...] = self.sobel_0
#        top[2].data[...] = self.sobel_1
#        top[3].data[...] = self.sobel_2
#        top[4].data[...] = self.sobel_3
#        top[5].data[...] = self.sobel_4
#        top[6].data[...] = self.sobel_5
        top[1].data[...] = self.labels
        
    def backward(self, bottom, top):
        pass