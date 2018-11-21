import caffe
import numpy as np
import global_var as GV
from scipy.misc import imresize
import random
import matplotlib.pyplot as plt
from PIL import Image

GV.c = np.zeros([3,3])
GV.norm_loss = np.zeros([3, 3])

class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def smooth_L1_loss(self, a, b):
        raw_loss = (a - b)
#        GV.a = a
#        GV.b = b
        gradient = np.ones_like(a, dtype = np.float32)
        gradient[np.where(raw_loss < 0)] = -1
        index = np.where(np.abs(raw_loss) < 1)
        loss = np.abs(raw_loss) - 0.5
        loss[index] = 0.5 * ((loss[index] + 0.5)** 2)
        gradient[index] = raw_loss[index]
        return loss, gradient
        
    def softmax_loss(self, a, b): #a = data, b = labels
        max_value = a.max(axis = 1)
        a = a - np.insert(max_value[:, np.newaxis], 1, values = [max_value for c in range(self.num_classes - 1)], axis = 1)
        a = np.exp(a)
        sum_value = a.sum(axis = 1)
        a = a / np.insert(sum_value[:, np.newaxis], 1, values = [sum_value for c in range(self.num_classes - 1)], axis = 1)

        a[np.where(a < 1e-10)] = 1e-10
        loss = - np.log(a) * b
        gradient = a - b
        GV.norm_loss = a
        return a, loss, gradient

    def sigmoid_crossentropyx_loss(self, a, b): #a = data, b = labels
        loss = np.log(1 + np.exp(a)) - a * b
        index = np.where(a >= 0)
        loss[index] = np.log(1 + np.exp(- a[index])) - (b[index] - 1) * a[index]
        gradient = (1 / (1 + np.exp( - a))) - b
        out = (1 / (1 + np.exp( - a)))
        return out, loss, gradient
    
    def hard_negative_mining(self, result_loss, labels):
        
        if result_loss.shape != labels.shape:
            print result_loss.shape, labels.shape
            raise Exception('Both inputs shape must be same')
        num_cases, channels, height, width = labels.shape
        pos_mask = np.zeros_like(labels)
        neg_mask = np.zeros_like(labels)
        
        for i in range(num_cases):
            pos_or_neg = 1
            pos_num = np.sum(labels[i])
            neg_num = np.sum(1 - labels[i])
            if pos_num > neg_num:
                target_labels = labels[i]
                target_num = neg_num
            else:
                target_num = pos_num
                target_labels = 1 - labels[i]
                pos_or_neg = 0
#                print 'neg is more'
            mask_0 = 1 - target_labels
            target_loss = target_labels * result_loss[i]
            target_loss[np.where(target_loss == 0)] = -100
            flatten_loss = target_loss.flatten()
            mask = np.zeros_like(flatten_loss)
            zip_loss = zip(flatten_loss, [p for p in range(len(flatten_loss))])
            zip_loss.sort(reverse = True)
            for j in range(target_num):
                mask[zip_loss[j][1]] = 1
            mask = mask.reshape(channels, height, width)
#            print 'hello',i
            if pos_or_neg == 1:
                pos_mask[i] = mask
                neg_mask[i] = mask_0
            else:
                pos_mask[i] = mask_0
                neg_mask[i] = mask
            
        return pos_mask, neg_mask
    
#    def hard_negative_mining(self, result_loss, labels):
#        select_channel = 0
#        samples_num = []
#        for i in range(self.num_classes):
#            samples_num.append(labels[:, i].sum())
#        
#        select_channel = np.argmin(samples_num)
#        samples_num = samples_num[select_channel]
#        if samples_num <100:
#            samples_num = 100
#        pos_matrix = labels[:, select_channel]
#        neg_samples_matrix = np.zeros_like(pos_matrix).flatten()
#        
#        tmp_loss = np.zeros_like(pos_matrix)
#        tmp_loss = np.sum(result_loss, axis = 1)
#            
#        flatten_pos_matrix = pos_matrix.flatten()
#        
#        flatten_loss = tmp_loss.flatten()
#        num_cases = flatten_loss.shape[0]
#        flatten_loss = zip(flatten_loss, [i for i in range(num_cases)])
#        flatten_loss.sort(reverse = True)
#        
#        neg_samples_index = np.zeros(samples_num)
#        
#        count = 0
#        i = 0
#        while count < samples_num:
#            if flatten_pos_matrix[flatten_loss[i][1]] == 0:
#                neg_samples_index[count] = flatten_loss[i][1]
#                count += 1
#                i += 1
#            else:
#                i += 1
#        for i in range(samples_num):
#            neg_samples_matrix[int(neg_samples_index[i])] = 1
#            
#        neg_samples_matrix = neg_samples_matrix.reshape(pos_matrix.shape)
#        
#        totoal_samples_matrix = neg_samples_matrix + pos_matrix
#        totoal_samples_matrix = np.tile(totoal_samples_matrix[:, np.newaxis], [1, self.num_classes, 1, 1])
#        return totoal_samples_matrix
    
    def flip_lr_ud(self, a, num, restore_flag = False):
        if restore_flag == False:
            if num / 2 == 1:
                a = a.transpose(2, 3, 0, 1)
                a = np.flipud(a)
                a = a.transpose(2, 3, 0, 1)
            if num % 2 == 1:
                a = a.transpose(2, 3, 0, 1)
                a = np.fliplr(a)
                a = a.transpose(2, 3, 0, 1)
        else:
            if num % 2 == 1:
                a = a.transpose(2, 3, 0, 1)
                a = np.fliplr(a)
                a = a.transpose(2, 3, 0, 1)
            if num / 2 == 1:
                a = a.transpose(2, 3, 0, 1)
                a = np.flipud(a)
                a = a.transpose(2, 3, 0, 1)
        return a
        
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.crop_size = params['crop_size']
        self.batches = params['batches']
        self.sup_threshold = params['sup_threshold']
        self.inf_threshold = params['inf_threshold']
        self.num_classes = 2

    def reshape(self, bottom, top):
        self.scale0_data = bottom[0].data
        GV.xxx = bottom
#        das
        saliency_map = bottom[1].data[:,0:2]
        self.labels = bottom[2].data
        self.num_cases, self.channels, self.height, self.width = self.scale0_data.shape
        saliency_labels = self.labels
        saliency_labels = np.concatenate((1 - self.labels, self.labels), axis = 1)
        saliency_map, self.sal_loss, self.sal_diff = self.softmax_loss(saliency_map, saliency_labels)
        GV.sal = saliency_map[0, 1]
        mod = np.sqrt(np.sum(self.scale0_data ** 2, axis = 1))
        GV.before = np.array(self.scale0_data[0].transpose(1,2,0))
        self.scale0_data = self.scale0_data.transpose(1, 0, 2, 3) / mod
        self.scale0_data = self.scale0_data.transpose(1, 0, 2, 3)
        GV.after = np.array(self.scale0_data[0].transpose(1,2,0))
        
        self.scale0_labels = np.zeros([self.num_cases, 1, self.height, self.width])
        for i in range(self.num_cases):
            tmp = imresize(np.array(self.labels[i, 0], dtype = np.uint8), [self.height, self.width])
            self.scale0_labels[i] = tmp[np.newaxis]
        self.scale0_labels = np.array(self.scale0_labels, dtype = np.float32)
        
#        self.scale0_data = np.tile(self.scale0_labels, (1, self.channels, 1, 1))
        num_foreground_example = np.sum(self.scale0_labels.reshape(self.num_cases, self.height * self.width), axis = 1)
        num_foreground_example[np.where(num_foreground_example == 0)] = 1
#        print 'fore_num', num_foreground_example
        tmp_foreground_example = self.scale0_data * \
                                    np.tile(self.scale0_labels, (1, self.channels, 1, 1))
        tmp_foreground_example = tmp_foreground_example.reshape(self.num_cases, self.channels, self.height * self.width)
        tmp_foreground_example = tmp_foreground_example.sum(axis = 2).transpose(1, 0) * 1. / num_foreground_example
        tmp_foreground_example = tmp_foreground_example.transpose(1, 0)

#        tmp_foreground_example = tmp_foreground_example.mean(axis = 0)
        
        num_background_example = np.sum(1 - self.scale0_labels.reshape(self.num_cases, self.height * self.width), axis = 1)
        num_background_example[np.where(num_background_example == 0)] = 1
#        print 'back_num', num_background_example
        tmp_background_example = self.scale0_data * \
                                    np.tile(1 - self.scale0_labels, (1, self.channels, 1, 1))
        tmp_background_example = tmp_background_example.reshape(self.num_cases, self.channels, self.height * self.width)
        tmp_background_example = tmp_background_example.sum(axis = 2).transpose(1, 0) * 1. / num_background_example
        tmp_background_example = tmp_background_example.transpose(1, 0)
#        tmp_background_example = tmp_background_example.mean(axis = 0)
#        if np.isnan(tmp_background_example.max()):
#            dasda
        tmp_scale0_data = self.scale0_data.transpose(2, 3, 0, 1)
        tmp_scale0_labels = self.scale0_labels.transpose(2, 3, 0, 1)
        
        foreground_loss = ((tmp_scale0_data - tmp_foreground_example) ** 2 - (tmp_scale0_data - tmp_background_example) ** 2) * \
                                np.tile(tmp_scale0_labels, (1, 1, 1, self.channels))
        
        foreground_loss = foreground_loss.sum(axis = 3)
        GV.foreground_loss = foreground_loss
        
        background_loss = ((tmp_scale0_data - tmp_background_example) ** 2 - (tmp_scale0_data - tmp_foreground_example) ** 2) * \
                                np.tile(1 - tmp_scale0_labels, (1, 1, 1, self.channels))
        
        background_loss = background_loss.sum(axis = 3)
        GV.background_loss = background_loss
        
        total_loss = foreground_loss + background_loss
        total_loss = total_loss[:,:,:,np.newaxis].transpose(2, 3, 0, 1)
#        total_loss = total_loss + self.sal_loss[:,:, np.newaxis].sum(axis = 1)
        
        tmp_scale0_labels = np.array(self.scale0_labels)
        
#        mask = np.ones_like(tmp_scale0_labels)
#        mask[np.where(total_loss[:,:,:,np.newaxis] <= -1)] = 0
#        tmp_scale0_labels[np.where(total_loss[:,:,:,np.newaxis].transpose(2, 3, 0, 1) <= -1)] = 0
        GV.tmp_scale0_labels = tmp_scale0_labels.transpose(2, 3, 0, 1)
        pos_mask, neg_mask = self.hard_negative_mining(total_loss, tmp_scale0_labels)
#        print pos_mask.shape
#        sda
#        pos_mask = ((2. * ((tmp_scale0_data - tmp_foreground_example) ** 2).sum(axis = 3)) > ((tmp_scale0_data - tmp_background_example) ** 2).sum(axis = 3)) * 1. * \
#                                tmp_scale0_labels[:, :, :, 0]
#        neg_mask = ((2. * ((tmp_scale0_data - tmp_background_example) ** 2).sum(axis = 3)) > ((tmp_scale0_data - tmp_foreground_example) ** 2).sum(axis = 3)) * 1. * \
#                                (1 - tmp_scale0_labels)[:, :, :, 0]
#        pos_mask = np.tile(pos_mask[:,:,:,np.newaxis], (1, 1, 1, self.channels))
#        neg_mask = np.tile(neg_mask[:,:,:,np.newaxis], (1, 1, 1, self.channels))

                                
#        pos_mask = tmp_scale0_labels * mask
#        neg_mask = (1 - tmp_scale0_labels) * mask
#        neg_mask = neg_mask * pos_mask.sum() * 1. / neg_mask.sum()
        
        pos_mask = pos_mask.transpose(2, 3, 0, 1)
        neg_mask = neg_mask.transpose(2, 3, 0, 1)
        
#        pos_mask = np.tile(pos_mask, (1, 1, 1, self.channels)) * np.tile(tmp_scale0_labels, (1, 1, 1, self.channels))
#        neg_mask = np.tile(neg_mask, (1, 1, 1, self.channels)) * np.tile(1 - tmp_scale0_labels, (1, 1, 1, self.channels))
#        print pos_mask[:,:,0,0].sum(), neg_mask[:,:,0,0].sum()
#        select_background_mask[np.where(background_loss > self.sup_threshold)] = 0
        foreground_diff = (tmp_background_example - tmp_foreground_example) * pos_mask
                                
        background_diff = (tmp_foreground_example - tmp_background_example) * neg_mask 
        
        totoal_diff = foreground_diff + background_diff
        totoal_diff = totoal_diff.transpose(2, 3, 0, 1)
        GV.total_loss = total_loss
        GV.pos_mask = pos_mask
        GV.neg_mask = neg_mask
#        GV.background_loss = background_loss
#        GV.foreground_loss = foreground_loss
        GV.foreground_diff = foreground_diff.transpose(2, 3, 0, 1)
        GV.background_diff = background_diff.transpose(2, 3, 0, 1)
        GV.scale0_data = self.scale0_data
        GV.scale0_labels = self.scale0_labels
        GV.tmp_foreground_example = tmp_foreground_example
        GV.tmp_background_example = tmp_background_example
        
        self.diff = totoal_diff
        
#        self.back_loss = np.sum((tmp_scale0_data - tmp_background_example) ** 2)  *  np.tile(1 - tmp_scale0_labels, (1, 1, 1, self.channels))
#        self.loss = np.sum((tmp_foreground_example - tmp_background_example) ** 2) / self.num_cases
        self.loss = np.array([total_loss.mean() + 4, self.sal_loss.mean()]) #, 
#        GV.sal_loss = np.array(self.sal_loss)
        GV.metric_loss = np.array(total_loss)
        
        if GV.phase == 'test' and GV.test_nums == 0:
            GV.test_nums = 1
        if GV.phase == 'test':
            EPSILON = 1e-8
            
#            saliency_map = self.scale0_data
            tmp = np.array(saliency_map[:, 1, np.newaxis])
            bounduary = 20
            tmp[:,:,bounduary:-bounduary, bounduary:-bounduary] = 1
            pos_index = np.where(tmp < 0.5)
            
            max_sample_num = 100
            if pos_index[0].__len__() > max_sample_num:
                sample_num = max_sample_num
            else:
                sample_num = pos_index[0].__len__()
            pos_sample_index = random.sample(np.arange(pos_index[0].__len__()), sample_num)
            
            sample1 = [0, 0]
            sample2 = [0, 0]
            for i in range(len(pos_sample_index)):
                
                tmp = self.scale0_data[0, :, pos_index[2][pos_sample_index[i]], pos_index[3][pos_sample_index[i]]]
                if i == 0:
                    sample1[0] += 1
                    sample1[1] += tmp
                else:
                    gap = np.sum((tmp - sample1[1] / sample1[0]) ** 2)
                    gap1 = 2
                    if sample2[0] != 0:
                        gap1 = np.sum((tmp - sample2[1] / sample2[0]) ** 2)
                    print gap, gap1
                    if gap > gap1:
                        sample2[0] += 1
                        sample2[1] += tmp
                    else:
                        sample1[0] += 1
                        sample1[1] += tmp
            if sample1[0] >= sample2[0]:
                if sample1[0] == 0:
                    sample1[0] = 1
                background_sample = sample1[1] / sample1[0]
            else:
                background_sample = sample2[1] / sample2[0]
            print sample1[0], sample2[0]
            
            tmp = self.scale0_data[0]
            GV.a = np.array(tmp)
            GV.final = np.sum((tmp.transpose(1, 2, 0) - background_sample) ** 2, axis = 2)
#            GV.w = np.array(GV.final)
#            print 'aaa',GV.final.max()
#            GV.final[np.where(GV.final < 2 )] = 0
            result = GV.final
#            if GV.final.max() != 0:
#                GV.final = (GV.final - GV.final.min()) * 1. / (GV.final.max() - GV.final.min()) * 255.0
            result = (result - np.min(result) + EPSILON) / (np.max(result) - np.min(result) + EPSILON) * 255
            
            scale0_labels = GV.scale0_labels[0] * 255
            plt.figure(1)
            plt.subplot(141)
            plt.imshow(np.array(GV.image, dtype = np.uint8))
            plt.subplot(142)
            plt.imshow(np.array(scale0_labels[0], dtype = np.uint8))
            plt.subplot(143)
            plt.imshow(np.array(result, dtype = np.uint8))
            plt.subplot(144)
            sal = saliency_map[0, 1]
            
#            GV.w = np.array(result)
#            GV.r = background_sample
            
#            GV.n = np.array(result)
#            if result.mean() > 180:
                
#                dsadaf
            sal = (sal - np.min(sal) + EPSILON) / (np.max(sal) - np.min(sal) + EPSILON) * 255
            plt.imshow(np.array(sal, dtype = np.uint8))
            img = Image.fromarray(np.array(GV.image, dtype = np.uint8))
            img.save(GV.target_data_dir + '/' + GV.data_name + '_images.jpg')
            print 'xxxxxxx'
            print GV.target_data_dir + '/' + GV.data_name + '_images.jpg'
            print GV.data_name
#            dasa
            img = Image.fromarray(np.array(scale0_labels[0], dtype = np.uint8))
            img.save(GV.target_data_dir + '/' + GV.data_name + '_labels.jpg')
            img = Image.fromarray(np.array(result, dtype = np.uint8))
            img.save(GV.target_data_dir + '/' + GV.data_name + '_metric_result.jpg')
            img = Image.fromarray(np.array(sal, dtype = np.uint8))
            img.save(GV.target_data_dir+ '/' + GV.data_name + '_sal_result.jpg')
            GV.test_nums += 1 
        top[0].reshape(*self.loss.shape)
        
    def forward(self, bottom, top):
        top[0].data[...] = self.loss
        
    def backward(self, top, propagate_down, bottom):
        GV.diff1 = self.diff / bottom[0].data.size
#        GV.diff2 = self.sal_diff / bottom[0].data.size
        bottom[0].diff[...] = self.diff / bottom[0].count * bottom[0].num#* GV.lr * 10
#        bottom[0].diff[...] = np.zeros_like([bottom[0].data])
        bottom[1].diff[...] = self.sal_diff / bottom[1].count * bottom[1].num
#        bottom[1].diff[...] = np.zeros_like([bottom[1].data])

        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        