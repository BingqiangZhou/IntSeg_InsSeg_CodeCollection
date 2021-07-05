import matplotlib
matplotlib.use("Pdf")

import os
import sys

import caffe
import surgery, score

import numpy as np

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

caffe.set_device(1)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

solver = caffe.SGDSolver('./solvers/solver_tslfn_32s.prototxt')

weights = './pretrained_weights/ilsvrc/VGG-ILSVRC-16-fcn.caffemodel'
base_net = caffe.Net('./pretrained_weights/ilsvrc/VGG-ILSVRC-16-fcn.prototxt', './pretrained_weights/ilsvrc/VGG-ILSVRC-16-fcn.caffemodel',caffe.TEST)

# init

base_layers = [k for k in base_net.params.keys()]
               
for l in base_layers:
    if l=='conv1_1': #for the first convolutional layer
        print 'processing: '+ l + ', ' + l + '_fea'
              
        solver.net.params[l][0].data[:, :3] = base_net.params[l][0].data
        solver.net.params[l][1].data[...] = base_net.params[l][1].data
                
        solver.net.params[l+'_fea'][0].data[:, 0] = np.mean(base_net.params['conv1_1'][0].data, axis=1)
        solver.net.params[l+'_fea'][0].data[:, 1] = np.mean(base_net.params['conv1_1'][0].data, axis=1)
        solver.net.params[l+'_fea'][1].data[...] = base_net.params[l][1].data
    elif l=='conv5_1':
        print 'processing: '+ l
        solver.net.params[l][0].data[:, :512] = base_net.params[l][0].data
        solver.net.params[l][0].data[:, 512:] = base_net.params[l][0].data

        solver.net.params[l][1].data[...] = base_net.params[l][1].data
    elif l=='fc8-conv':
        print 'dropping: '+ l
    else:
        print 'copying: '+ l + ' -> ' + l
        solver.net.params[l][0].data[...] = base_net.params[l][0].data
        solver.net.params[l][1].data[...] = base_net.params[l][1].data
        if l+'_fea' in solver.net.params:
            print 'copying: '+ l + ' -> ' + l + '_fea'
            solver.net.params[l+'_fea'][0].data[...] = base_net.params[l][0].data
            solver.net.params[l+'_fea'][1].data[...] = base_net.params[l][1].data

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

del base_net

#scoring
val = np.loadtxt('./data/example_data/voc/imageList/val.txt', dtype=str)

for _ in range(2000):
    solver.step(5000)
    score.seg_tests(solver, False, val, layer='score',saveFilePath='./snapshots/snapshot_tslfn_32s/rstSnapshot.txt')