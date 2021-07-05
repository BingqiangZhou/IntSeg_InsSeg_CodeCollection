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

weights = './pretrained_weights/tslfn_8s/train_iter_950000.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(1)
#caffe.set_mode_cpu()

solver = caffe.SGDSolver('./solvers/solver_fctsfn.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

#solver.net.forward()

#print np.sum(np.abs(solver.net.blobs['fea_comb_2_up'].data))

# scoring
val = np.loadtxt('./data/example_data/voc/imageList/val.txt', dtype=str)

# solver.step(1)

for _ in range(2000):
    #solver.step(4000)
    #solver.step(50)
    solver.step(100)
    #solver.step(1000)
    score.seg_tests(solver, False, val, layer='score_f',saveFilePath='./snapshots/snapshot_fctsfn/rstSnapshot.txt')
