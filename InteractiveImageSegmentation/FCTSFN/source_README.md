# FCTSFN

This repository includes the [caffe][1] model and codes for the following paper on interactive image segmentation:

Y.Hu, A. Soltoggio, R. Lock and S. Carter. A Fully Convolutional Two-Stream Fusion Network for Interactive Image Segmentation. Neural Networks, vol.109, pp.31-42, 2019. 

The codes are tested with Python 2.7 and Ubuntu 16.04.

## Example run

Run `run_example.py` on an example to use the model file and pre-trained weights.

## Model training

See `./data/example_data/voc` an example of the way to organize training data (note that this is not the full data; see above paper on more detailed information on the data used in the paper).

Run `solve_32s.py`, `solve_16s.py`, `solve_8s.py` for the training of TSLFN subnet in FCTSFN from stride 32 to stride 8.

Run `solve_fctsfn.py` for the training of MSRN subnet in FCTSFN.

## Acknowledgement

Part of codes are modified from the codes in [FCN][2] repository.

[1]:https://github.com/BVLC/caffe "caffe"

[2]:https://github.com/shelhamer/fcn.berkeleyvision.org "FCN"
