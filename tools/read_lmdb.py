import lmdb
import numpy as np
import caffe

#file_path = '/home/huangjb/deep-residual-networks-master/data/cifar10/cifar10_train_lmdb/'


def read_lmdb(file_path):
    lmdb_env = lmdb.open(file_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    lmdb_cursor.next()
    value = lmdb_cursor.value()
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    channels = datum.channels
    height = datum.height
    width = datum.width
    data = np.zeros((1, channels, height, width), np.uint8)
    labels = np.zeros(1, np.uint8)
    i = 0
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        flat_x = np.fromstring(datum.data, dtype = np.int8)
        x = flat_x.reshape(1, datum.channels, datum.height, datum.width)
        y = np.array([datum.label], dtype = np.uint8)
        print 'key:',key, datum.channels, datum.height, datum.width, 'label:', y
        data = np.concatenate((data, x))
        labels = np.concatenate((labels, y))
#        i += 1
        if i == 10:
            break
    data = np.delete(data, 0, 0)
    labels = np.delete(labels, 0)
    return data, labels
    #label = datum.label
    #data = caffe.io.datum_to_array(datum)
    #for l, d in zip(label, data):
        #print l, d
    

#data1, label1 = read_lmdb(file_path)

