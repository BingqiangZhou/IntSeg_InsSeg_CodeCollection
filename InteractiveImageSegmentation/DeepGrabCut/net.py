import cv2 as cv
import numpy as np
import torch
from torch.nn.functional import interpolate

import networks.deeplab_resnet as resnet

class DeepGrabCut():
    def __init__(self, weights_path="./models/deepgc_pascal_epoch-99.pth", num_device=0, threshold=0.8):
        
        device_str = "cpu" if num_device < 0 else f"cuda:{num_device}"
        self.device = torch.device(device_str)
        self.threshold = threshold
        
        #  Create the network and load the weights
        self.net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
        print("Initializing weights from: {}".format(weights_path))

        self.net.load_state_dict(torch.load(weights_path))
        self.net.eval()
        self.net.to(self.device)

    def __bbox_to_binary_image__(self, size, bbox):
        binary = np.zeros(size, dtype=np.uint8)
        minx, miny, maxx, maxy = bbox
        binary[miny, minx:maxx] = 1
        binary[maxy, minx:maxx] = 1
        binary[miny:maxy, minx] = 1
        binary[miny:maxy, maxx] = 1
        return binary

    def predict(self, image, bbox):
        """
            image: RGB image,  np.ndarray
            bbox: [minx, miny, maxx, maxy], list or tuple

        """
        h, w = image.shape[:2]
        
        minx, miny, maxx, maxy = bbox
   
        bbox_binary_image = self.__bbox_to_binary_image__(image.shape[:2], bbox)
        bbox_binary_image = (bbox_binary_image == 0).astype(np.uint8)
        
        tmp_ = np.zeros((h, w), dtype=np.int8)
        tmp_[miny+1:maxy-1, minx+1:maxx-1] = -1  # pixel inside bounding box
        tmp_[tmp_ == 0] = 1  # pixel on and outside bounding box

        dismap = cv.distanceTransform(bbox_binary_image, cv.DIST_L2, cv.DIST_MASK_PRECISE)  # compute distance inside and outside bounding box
        dismap = tmp_ * dismap + 128
        dismap[dismap > 255] = 255
        dismap[dismap < 0] = 0
        cv.imshow("dismap", dismap.astype(np.uint8))
        
        # dismap = utils.fixed_resize(dismap, (450, 450)).astype(np.uint8)
        dismap = np.expand_dims(dismap, axis=-1)
        # input_image = utils.fixed_resize(image, (450, 450)).astype(np.uint8)
        # bbox_binary_image = utils.fixed_resize(bbox_binary_image, self.fix_size).astype(np.uint8)

        merge_input = np.concatenate((image, dismap), axis=2).astype(np.float32)
        inputs = torch.from_numpy(merge_input.transpose((2, 0, 1))[np.newaxis, ...])

        # Run a forward pass
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.net.forward(inputs) # stride = 8
            outputs = interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
            outputs = outputs.cpu()

        prediction = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
        # fixed: RuntimeWarning: overflow encountered in exp
        prediction[prediction < -20] = -20 # exp(20)=485165195.4097903
        prediction = 1 / (1 + np.exp(-prediction))
        prediction = np.squeeze(prediction)
        prediction = (prediction > self.threshold).astype(np.uint8)
        return prediction

# from PIL import Image
# import cv2 as cv

# net = DeepGrabCut()

# image = np.array(Image.open('./ims/2007_000039.jpg'))
# cv.imshow("image", image)
# h, w = image.shape[:2]
# result = net.predict(image, [50, 50, h- 50, w -50])
# cv.imshow("out", result*255)
# cv.waitKey(0)




