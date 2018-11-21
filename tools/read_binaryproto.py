
import os.path as osp
import caffe
#import sys

this_dir = osp.dirname(__file__)
#mean_file_path = osp.join(this_dir,'..', 'data', 'cifar10','mean.binaryproto')


def read_binaryproto(file_path):
   data_blob = caffe.proto.caffe_pb2.BlobProto()
   data_blob.ParseFromString(open(file_path, 'rb').read())
   data = caffe.io.blobproto_to_array(data_blob)
   return data

#data = read_binaryproto(mean_file_path)