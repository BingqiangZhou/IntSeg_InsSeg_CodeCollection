import caffe

import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt

import random

class VOCSegClickPatternDataLayer(caffe.Layer):
    """
    Modified from FCN codes: https://github.com/shelhamer/fcn.berkeleyvision.org
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract (note: it is 5-element tuple with 3 mean for RGB channels and 2 means for positive and negative click maps)
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        
        """
        # config
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir']
        self.split = params['split']
        self.tops = params['tops']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.scale = params.get('scale', 1.0)
	self.cntSmp=0
        self.totalSmp=91320
        
        # store top data for reshape + forward
        self.data = {}
        
        self.idxMap={}
        self.idxMap['image']=0
        self.idxMap['click']=3
        
        #self.isTest=True
        self.isTest=False
        
        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/imageList/{}.txt'.format(self.voc_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit (1 is the batch dim)
        for i, t in enumerate(self.tops):
            self.data[t] = self.load(t, self.indices[self.idx])
            top[i].reshape(1, *self.data[t].shape)


    def forward(self, bottom, top):
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
	self.cntSmp = (self.cntSmp+1) % self.totalSmp
        if self.cntSmp == self.totalSmp-1:
            self.seed=self.seed+100
            random.seed(self.seed)

    def backward(self, top, propagate_down, bottom):
        pass
    
    def load(self, top, idx):
        if top == 'image':
            return self.load_img(idx)
        elif top == 'click':
            return self.load_click(idx) 
        elif top == 'label':
            return self.load_label(idx)
        else:
            raise Exception("Unknown output type: {}".format(top))

    def load_img(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #names are separated by ','
        tmpPosSep1=idx.find(',')
        tmpImName=idx[:tmpPosSep1]
        #load image
 
        im = Image.open('{}/images/{}.tiff'.format(self.voc_dir, tmpImName))
        if self.isTest:
            print 'load {}/images/{}.tiff'.format(self.voc_dir, tmpImName)
        
        tmpIm=np.array(im, dtype=np.float32)
        if self.isTest:
            plt.figure();
            plt.imshow(tmpIm.astype(np.uint8))
        tmpIm=tmpIm[:,:,::-1]
        in_= np.array(tmpIm, dtype=np.float32)
        in_ -= self.mean[self.idxMap['image']:self.idxMap['image']+3]
        in_ = in_ * self.scale;
        in_ = in_.transpose((2,0,1))
        if self.isTest:
            print '****'
            print np.mean(tmpIm[:,:,0]-in_[0])
            print '****'
            print np.mean(tmpIm[:,:,1]-in_[1])
            print '****'
            print np.mean(tmpIm[:,:,2]-in_[2])
            print '****'
        del tmpIm;
        return in_
    
    def load_click(self, idx):       
        #names are separated by ','
        tmpPosSep1=idx.find(',')
        tmpImName=idx[:tmpPosSep1]
        tmpPosSep2=idx.find(',',tmpPosSep1+1)
        tmpPosFeaName=idx[tmpPosSep1+1:tmpPosSep2]
        tmpPosSep3=idx.find(',',tmpPosSep2+1)
        tmpNegFeaName=idx[tmpPosSep2+1:tmpPosSep3]
        #load image

        #load positive click pattern
        imPosClick=Image.open('{}/clickPattern/{}.png'.format(self.voc_dir, tmpPosFeaName))
        if self.isTest:
            print 'load {}/clickPattern/{}.png'.format(self.voc_dir, tmpPosFeaName)
        #load negative click pattern
        imNegClick=Image.open('{}/clickPattern/{}.png'.format(self.voc_dir, tmpNegFeaName))
        if self.isTest:
            print 'load {}/clickPattern/{}.png'.format(self.voc_dir, tmpNegFeaName)

        tmpImPosClick=np.array(imPosClick, dtype=np.float32)
        tmpImNegClick=np.array(imNegClick, dtype=np.float32)
        if self.isTest:
            plt.figure();
            plt.imshow(tmpImPosClick,cmap='gray')
            plt.figure();
            plt.imshow(tmpImNegClick,cmap='gray')
        in_=np.dstack((tmpImPosClick,tmpImNegClick))        
        in_ -= self.mean[self.idxMap['click']:self.idxMap['click']+2]
        in_ = in_ * self.scale;
        in_ = in_.transpose((2,0,1))
        if self.isTest:
            print '****'
            print np.mean(tmpImPosClick-in_[0])
            print '****'
            print np.mean(tmpImNegClick-in_[1])
            print '****'
        del tmpImPosClick;
        del tmpImNegClick;
        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        tmpPosSep1=idx.find(',')
        tmpPosSep2=idx.find(',',tmpPosSep1+1)
        tmpPosSep3=idx.find(',',tmpPosSep2+1)
        gtImgName=idx[tmpPosSep3+1:];

        im = Image.open('{}/gt/{}.png'.format(self.voc_dir, gtImgName))
        if self.isTest:
            print 'load {}/gt/{}.png'.format(self.voc_dir, gtImgName)

        label = np.array(im, dtype=np.uint8)
        label[np.where(label==255)]=1
        if self.isTest:
            plt.figure();
            plt.imshow(label,cmap='gray')
        label = label[np.newaxis, ...]
        return label