#This code is from FCN github: https://github.com/shelhamer/fcn.berkeleyvision.org

from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
#import matplotlib.pyplot as plt

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)
        
        #print net.blobs[layer].data[0].argmax(0)
        #plt.figure()
        #plt.imshow(net.blobs[layer].data[0].argmax(axis=0))
        #print np.sum(np.absolute(net.blobs[layer].data[0]))
        #print net.blobs[layer].data[0].shape
        #print np.mean(net.blobs[layer].data[0][0,:,:])
        #print np.mean(net.blobs[layer].data[0][1,:,:])
        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    #print hist;
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label', saveFilePath=None):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)    
    #solver.test_nets[0].copy_from(saveFilePath[:-15]+'train_iter_'+str(solver.iter)+'.caffemodel')    
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt, saveFilePath)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label',saveFilePath=None):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    #print hist[0,0]
    #print hist[0,1]
    #print hist[1,0]
    #print hist[1,1]
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    #print np.diag(hist).sum()
    #print hist.sum()
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    print np.diag(hist)
    print hist.sum(1)
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    
    if saveFilePath is not None:
        with open(saveFilePath,'ab+') as f:
            acc = np.diag(hist).sum() / hist.sum()
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' loss ' + str(loss) + os.linesep
            f.write(tmpStr)
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' overall accuracy ' + str(acc) + os.linesep
            f.write(tmpStr)
            acc = np.diag(hist) / hist.sum(1)
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' mean accuracy ' + str(np.nanmean(acc)) + os.linesep
            f.write(tmpStr)
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' mean IU ' + str(np.nanmean(iu)) + os.linesep
            f.write(tmpStr)
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' fwavacc ' + str((freq[freq > 0] * iu[freq > 0]).sum()) + os.linesep
            f.write(tmpStr)
            f.write('********************************' + os.linesep)
        f.close()
    return hist
    
def seg_tests_fg(solver, save_format, dataset, layer='score', gt='label', saveFilePath=None):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests_fg(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt, saveFilePath)

def do_seg_tests_fg(net, iter, save_format, dataset, layer='score', gt='label',saveFilePath=None):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)    
    print '>>>', datetime.now(), 'Iteration', iter, 'IU fg', iu[1]
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    
    if saveFilePath is not None:
        with open(saveFilePath,'ab+') as f:
            acc = np.diag(hist).sum() / hist.sum()
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' loss ' + str(loss) + os.linesep
            f.write(tmpStr)
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' overall accuracy ' + str(acc) + os.linesep
            f.write(tmpStr)
            acc = np.diag(hist) / hist.sum(1)
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' mean accuracy ' + str(np.nanmean(acc)) + os.linesep
            f.write(tmpStr)
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' mean IU ' + str(np.nanmean(iu)) + os.linesep
            f.write(tmpStr)
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' IU fg ' + str(iu[1]) + os.linesep
            f.write(tmpStr)
            tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(iter) + ' fwavacc ' + str((freq[freq > 0] * iu[freq > 0]).sum()) + os.linesep
            f.write(tmpStr)
            f.write('********************************' + os.linesep)
        f.close()
    return hist
