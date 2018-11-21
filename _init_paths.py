import os.path as osp
import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0,path)

this_dir = osp.dirname(__file__)

pycaffe_path = osp.join(this_dir, 'caffe', 'python')
add_path(pycaffe_path)

layers_path = osp.join(this_dir, 'layers')
add_path(layers_path)

tools_path = osp.join(this_dir, 'tools')
add_path(tools_path)

model_path = osp.join(this_dir, 'model')
add_path(model_path)




