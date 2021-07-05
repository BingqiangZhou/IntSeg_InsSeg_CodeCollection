
import caffe

import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt

import random

from skimage.transform import resize

import cv2

class VOCSegClickPatternBatchInputDataAugDataLayer(caffe.Layer):
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
        self.batch_size = 3
        self.sizeH = 240
        self.sizeW = 320
	self.isCrop = np.array([False,False,False])
        self.cropCenH = np.array([0,0,0])
        self.cropCenW = np.array([0,0,0])
        self.cropH = np.array([0,0,0])
        self.cropW = np.array([0,0,0])
        self.rotCenH = np.array([0,0,0])
        self.rotCenW = np.array([0,0,0])
	self.rotAngle = np.array([0,0,0])
        self.translateW = np.array([0,0,0])
        self.translateH = np.array([0,0,0])	
	self.isTransAug = np.array([False,False,False])
        self.isRotAug = np.array([False,False,False])

        # store top data for reshape + forward
        self.data = {}
	self.data['image']=np.zeros((self.batch_size,3,self.sizeH,self.sizeW))
        self.data['click']=np.zeros((self.batch_size,2,self.sizeH,self.sizeW))
        self.data['label']=np.zeros((self.batch_size,1,self.sizeH,self.sizeW))
        
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
        self.idx = list([0,0,0])
	
	self.totalSmp=len(self.indices)
	print 'total sample: '+str(self.totalSmp)

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False
            self.isTransAug = np.array([False,False,False])
            self.isRotAug = np.array([False,False,False])

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
	    for cnt_batch_idx in range(0,self.batch_size):
	        self.idx[cnt_batch_idx] = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
	# determine data augment/crop
	if 'train' in self.split:
	    for cnt in range(self.batch_size):
	        tmpLabel=self.load_label_ori(self.indices[self.idx[cnt]])
		tmpLabel=tmpLabel.transpose((1,2,0))[:,:,0]
	        my,mx=np.where(tmpLabel==1)
	        tmpRatio = float((np.amax(my)-np.amin(my))*(np.amax(mx)-np.amin(mx)))/float(self.sizeH*self.sizeW)
	        if tmpRatio<0.35:
		    self.isCrop[cnt]=True
     		    self.cropCenH[cnt]=np.mean(my)
		    self.cropCenW[cnt]=np.mean(mx)
		    self.cropH[cnt]=np.minimum(self.sizeH-np.amax(my),np.amin(my))/2+(np.amax(my)-np.amin(my))/2
		    self.cropW[cnt]=np.minimum(self.sizeW-np.amax(mx),np.amin(mx))/2+(np.amax(mx)-np.amin(mx))/2
		    x1=np.maximum(self.cropCenW[cnt]-self.cropW[cnt],0)
		    x2=np.minimum(self.cropCenW[cnt]+self.cropW[cnt],self.sizeW)
		    y1=np.maximum(self.cropCenH[cnt]-self.cropH[cnt],0)
		    y2=np.minimum(self.cropCenH[cnt]+self.cropH[cnt],self.sizeH)
		    tmpLabel_crop=tmpLabel[y1:y2,x1:x2]
		    my1,mx1=np.where(tmpLabel_crop==1)
		    self.rotCenH[cnt]=np.mean(my1)
		    self.rotCenW[cnt]=np.mean(mx1)
		    tmpBd_transH=np.minimum(tmpLabel_crop.shape[0]-np.amax(my1),np.amin(my1))/3
                    tmpBd_transW=np.minimum(tmpLabel_crop.shape[1]-np.amax(mx1),np.amin(mx1))/3		    
	        else:
		    self.isCrop[cnt]=False
		    self.rotCenH[cnt]=np.mean(my)
                    self.rotCenW[cnt]=np.mean(mx)
		    tmpBd_transH=np.minimum(self.sizeH-np.amax(my),np.amin(my))/3
		    tmpBd_transW=np.minimum(self.sizeW-np.amax(mx),np.amin(mx))/3
		#rotation aug
            	if random.uniform(0.0,1.0)>0.5:
		    self.isRotAug[cnt]=True
		    self.rotAngle[cnt]=random.uniform(-60.0,60.0)
	
		if random.uniform(0.0,1.0)>0.5:
		    self.isTransAug[cnt]=True
		    self.translateW[cnt]=random.randint(-tmpBd_transW,tmpBd_transW)
		    self.translateH[cnt]=random.randint(-tmpBd_transH,tmpBd_transH)

        # load data for tops and  reshape tops to fit (1 is the batch dim)
        for i, t in enumerate(self.tops):
	    for cnt in range(self.batch_size):
                self.data[t][cnt,:,:,:] = self.load(t, self.indices[self.idx[cnt]],cnt)
            top[i].reshape(*self.data[t].shape)


    def forward(self, bottom, top):
	#print self.isCrop
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]

        # pick next input
        if self.random:
	    for cnt_batch_idx in range(0,self.batch_size):
                self.idx[cnt_batch_idx] = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
	self.cntSmp = self.cntSmp+self.batch_size
        if self.cntSmp >= self.totalSmp:
	    self.cntSmp=0
            self.seed=1000+random.randint(0, 5000)
            random.seed(self.seed)
	#reset data augment/crop parameters
	self.isCrop = np.array([False,False,False])
        self.cropCenH = np.array([0,0,0])
        self.cropCenW = np.array([0,0,0])
        self.cropH = np.array([0,0,0])
        self.cropW = np.array([0,0,0])
        self.rotCenH = np.array([0,0,0])
        self.rotCenW = np.array([0,0,0])
  	self.rotAngle = np.array([0,0,0])
        self.translateW = np.array([0,0,0])
        self.translateH = np.array([0,0,0])	
	self.isTransAug = np.array([False,False,False])
	self.isRotAug = np.array([False,False,False])

    def backward(self, top, propagate_down, bottom):
        pass
    
    def load(self, top, idx, cnt):
        if top == 'image':
            return self.load_img(idx,cnt)
        elif top == 'click':
            return self.load_click(idx,cnt)    
        elif top == 'label':
            return self.load_label(idx,cnt)
        else:
            raise Exception("Unknown output type: {}".format(top))

    def load_img(self, idx, cnt_idx):
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
	tmpIm=resize(tmpIm, (self.sizeH, self.sizeW), clip=False, preserve_range=True)
	if self.isCrop[cnt_idx]:
	    x1=np.maximum(self.cropCenW[cnt_idx]-self.cropW[cnt_idx],0)
            x2=np.minimum(self.cropCenW[cnt_idx]+self.cropW[cnt_idx],self.sizeW)
            y1=np.maximum(self.cropCenH[cnt_idx]-self.cropH[cnt_idx],0)
            y2=np.minimum(self.cropCenH[cnt_idx]+self.cropH[cnt_idx],self.sizeH)
	    tmpIm=tmpIm[y1:y2,x1:x2]
	#rotation aug
	if self.isRotAug[cnt_idx]:
	    M = cv2.getRotationMatrix2D((self.rotCenW[cnt_idx],self.rotCenH[cnt_idx]),self.rotAngle[cnt_idx],1)
	    tmpIm = cv2.warpAffine(tmpIm,M,(tmpIm.shape[1],tmpIm.shape[0]))
        #translate aug
        if self.isTransAug[cnt_idx]:
            M = np.float32([[1,0,self.translateW[cnt_idx]],[0,1,self.translateH[cnt_idx]]])
	    tmpIm = cv2.warpAffine(tmpIm,M,(tmpIm.shape[1],tmpIm.shape[0]))
	if self.isCrop[cnt_idx]:
	    tmpIm=resize(tmpIm, (self.sizeH, self.sizeW), clip=False, preserve_range=True)
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
    
    def load_click(self, idx, cnt_idx):       
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

	tmpImPosClick=resize(tmpImPosClick, (self.sizeH, self.sizeW), clip=False, preserve_range=True)
        tmpImNegClick=resize(tmpImNegClick, (self.sizeH, self.sizeW), clip=False, preserve_range=True)

	if self.isCrop[cnt_idx]:
            x1=np.maximum(self.cropCenW[cnt_idx]-self.cropW[cnt_idx],0)
            x2=np.minimum(self.cropCenW[cnt_idx]+self.cropW[cnt_idx],self.sizeW)
            y1=np.maximum(self.cropCenH[cnt_idx]-self.cropH[cnt_idx],0)
            y2=np.minimum(self.cropCenH[cnt_idx]+self.cropH[cnt_idx],self.sizeH)
            tmpImPosClick=tmpImPosClick[y1:y2,x1:x2]
 	    tmpImNegClick=tmpImNegClick[y1:y2,x1:x2]
	#rotation aug
        if self.isRotAug[cnt_idx]:
            M = cv2.getRotationMatrix2D((self.rotCenW[cnt_idx],self.rotCenH[cnt_idx]),self.rotAngle[cnt_idx],1)
	    tmpImPosClick = cv2.warpAffine(tmpImPosClick,M,(tmpImPosClick.shape[1],tmpImPosClick.shape[0]))
            tmpImNegClick = cv2.warpAffine(tmpImNegClick,M,(tmpImNegClick.shape[1],tmpImNegClick.shape[0]))
        #translate aug
        if self.isTransAug[cnt_idx]:
            M = np.float32([[1,0,self.translateW[cnt_idx]],[0,1,self.translateH[cnt_idx]]])
	    tmpImPosClick = cv2.warpAffine(tmpImPosClick,M,(tmpImPosClick.shape[1],tmpImPosClick.shape[0]))
            tmpImNegClick = cv2.warpAffine(tmpImNegClick,M,(tmpImNegClick.shape[1],tmpImNegClick.shape[0]))
        if self.isCrop[cnt_idx]:
	    tmpImPosClick=resize(tmpImPosClick, (self.sizeH, self.sizeW), clip=False, preserve_range=True)
            tmpImNegClick=resize(tmpImNegClick, (self.sizeH, self.sizeW), clip=False, preserve_range=True)
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

    def load_label_ori(self, idx):
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
        label=resize(label, (self.sizeH, self.sizeW), clip=False, preserve_range=True)

        label[np.where(label!=0)]=1

        if self.isTest:
            plt.figure();
            plt.imshow(label,cmap='gray')
        label = label[np.newaxis, ...]
        return label

    def load_label(self, idx, cnt_idx):
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
	label=resize(label, (self.sizeH, self.sizeW), clip=False, preserve_range=True)

        label[np.where(label!=0)]=1

	if self.isCrop[cnt_idx]:
            x1=np.maximum(self.cropCenW[cnt_idx]-self.cropW[cnt_idx],0)
            x2=np.minimum(self.cropCenW[cnt_idx]+self.cropW[cnt_idx],self.sizeW)
            y1=np.maximum(self.cropCenH[cnt_idx]-self.cropH[cnt_idx],0)
            y2=np.minimum(self.cropCenH[cnt_idx]+self.cropH[cnt_idx],self.sizeH)
            label=label[y1:y2,x1:x2]
	#rotation aug
        if self.isRotAug[cnt_idx]:
            M = cv2.getRotationMatrix2D((self.rotCenW[cnt_idx],self.rotCenH[cnt_idx]),self.rotAngle[cnt_idx],1)
	    label = cv2.warpAffine(label,M,(label.shape[1],label.shape[0]))
            label[np.where(label!=0)]=1
        #translate aug
        if self.isTransAug[cnt_idx]:
            M = np.float32([[1,0,self.translateW[cnt_idx]],[0,1,self.translateH[cnt_idx]]])
	    label = cv2.warpAffine(label,M,(label.shape[1],label.shape[0]))
            label[np.where(label!=0)]=1
        if self.isCrop[cnt_idx]:
	    label=resize(label, (self.sizeH, self.sizeW), clip=False, preserve_range=True)
            label[np.where(label!=0)]=1
        if self.isTest:
            plt.figure();
            plt.imshow(label,cmap='gray')
        label = label[np.newaxis, ...]
        return label
