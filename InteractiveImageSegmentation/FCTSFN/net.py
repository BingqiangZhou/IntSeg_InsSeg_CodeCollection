import cv2 as cv
import numpy as np
import time
from scipy import ndimage

class FCTSFN:
    def __init__(self, prototxt_path='deploy_softmax.prototxt', 
                        caffeModel_path='./pretrained_weights/weights.caffemodel', 
                        use_gpu=True) -> None:
        self.data_mean = np.array([104.00699, 116.66877, 122.67892, 166.90557, 147.47697])
        self.net = cv.dnn.readNetFromCaffe(prototxt_path, caffeModel_path)
        if use_gpu and cv.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    
    def __pre_process(self, image, fg_interactive_map, bg_interactive_map):
        
        def bwdist(binary_mask):
            distance_map = ndimage.morphology.distance_transform_edt(1 - binary_mask)
            return distance_map 

        def dist_transform(interactive_map):
            distance_map = bwdist(interactive_map)
            distance_map[distance_map > 255] = 255
            return distance_map
        
        fg_distance_map = dist_transform(fg_interactive_map)
        bg_distance_map = dist_transform(bg_interactive_map)

        distance_maps = np.stack([fg_distance_map, bg_distance_map], axis=-1) # (h, w) * 2 -> (h, w, 2) 

        image = image - self.data_mean[:3]
        distance_maps = distance_maps - self.data_mean[3:]
        
        image = image.transpose((2, 0, 1))[np.newaxis, :, :, :]
        distance_maps = distance_maps.transpose((2, 0, 1))[np.newaxis, :, :, :]

        return image, distance_maps

    def __post_process(self, pred):
        # rst = 255*np.array(out[1,:,:],dtype=np.float32) # from `run_example.py`
        out = np.argmax(pred, axis=1) # (1, 2, h, w) -> (1, h, w)
        out = np.squeeze(out, axis=0) # -> (h, w) 
        return out

    def predict(self, image, fg_interactive_map, bg_interactive_map):
        image, distance_maps = self.__pre_process(image, fg_interactive_map, bg_interactive_map)

        self.net.setInput(image, "img")
        self.net.setInput(distance_maps, "click")

        out = self.net.forward()
        
        # print(out.shape)
        out = self.__post_process(out)

        return out


# size = (100, 100)
# image = np.random.rand(*size, 3)
# fg_dist_map = np.random.rand(*size)
# bg_dist_map = np.random.rand(*size)

# # prototxt_path = './models/val_fctsfn.prototxt'
# prototxt_path = 'deploy_softmax.prototxt'
# caffeModel_path = './pretrained_weights/weights.caffemodel'

# net = FCTSFN(prototxt_path, caffeModel_path, use_gpu=True)
# out = net.predict(image, fg_dist_map, bg_dist_map)