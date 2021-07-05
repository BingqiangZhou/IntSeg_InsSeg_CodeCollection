import matplotlib
matplotlib.use("Pdf")

import os
import sys

import caffe

import numpy as np
from PIL import Image
  
from util import getClickMap

#caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(1)

path_data='./data/example_data/test_example';
im_name='COCO_val2014_000000273493'
dataMean=np.array([104.00699, 116.66877, 122.67892, 166.90557, 147.47697],dtype=np.float64)
clk_pt_pos = np.array([[94,165],[98,196]])
clk_pt_neg = np.array([[216,138],[340,73]])

net = caffe.Net('./deploy_softmax.prototxt', './pretrained_weights/fctsfn/weights.caffemodel', caffe.TEST)

im = Image.open(os.path.join(path_data,"{}.png".format(im_name)))

tmpIm = np.array(im,dtype=np.float64)
tmpImPosClick = getClickMap(clk_pt_pos, tmpIm.shape[:2])
tmpImNegClick = getClickMap(clk_pt_neg, tmpIm.shape[:2])

data_im=np.array(tmpIm[:,:,::-1])        
data_im -= dataMean[:3]
data_im = data_im.transpose((2,0,1))
            
data_click=np.dstack((tmpImPosClick,tmpImNegClick))
data_click-=dataMean[3:]
data_click = data_click.transpose((2,0,1))

net.blobs['img'].reshape(1, *data_im.shape)
net.blobs['img'].data[...] = data_im

net.blobs['click'].reshape(1, *data_click.shape)
net.blobs['click'].data[...] = data_click

net.forward()

out = net.blobs['prob'].data[0]
rst = 255*np.array(out[1,:,:],dtype=np.float32)

im_save = Image.fromarray(rst.astype(np.uint8))
im_save.save(os.path.join(path_data,"{}_prob_map.png".format(im_name)),'PNG')
