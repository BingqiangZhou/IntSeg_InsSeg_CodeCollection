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

weights = './pretrained_weights/tslfn_16s/train_iter_300000.caffemodel'

# init
caffe.set_device(1)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

solver = caffe.SGDSolver('./solvers/solver_tslfn_8s.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('./data/example_data/voc/imageList/val.txt', dtype=str)

for _ in range(2000):
    solver.step(5000)
    score.seg_tests(solver, False, val, layer='score',saveFilePath='./snapshots/snapshot_tslfn_8s/rstSnapshot.txt')
