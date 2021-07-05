import cv2 as cv
import numpy as np
from scipy import ndimage

class BRS:
    def __init__(self, prototxt_path='./model/deploy.prototxt', 
                        caffeModel_path='./model/BRS_DenseNet.caffemodel', 
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
        
        self.img_size = image.shape[:2] # (h, w)

        # 网络输入大小为32的倍数即可，训练网络的时候设置的大小为480x480
        long_len = max(self.img_size)
        net_size = [480, 480]
        x_wholeLen = (self.img_size[1]*net_size[1]/long_len) # w
        y_wholeLen = (self.img_size[0]*net_size[0]/long_len) # h
        x_wholeLen = int(32*round(x_wholeLen/32))
        y_wholeLen = int(32*round(y_wholeLen/32))
        new_image = cv.resize(image, (x_wholeLen, y_wholeLen), interpolation=cv.INTER_CUBIC)
        new_image = new_image - [123.68, 116.779, 103.939] # [103.939, 116.779, 123.68]
        new_image = new_image*0.017

        new_fg_interactive_map = cv.resize(fg_interactive_map, (x_wholeLen, y_wholeLen), interpolation=cv.INTER_NEAREST)
        new_bg_interactive_map = cv.resize(bg_interactive_map , (x_wholeLen, y_wholeLen), interpolation=cv.INTER_NEAREST)
        fg_distance_map = 1 - dist_transform(new_fg_interactive_map) / 255.0
        bg_distance_map = 1 - dist_transform(new_bg_interactive_map) / 255.0

        distance_maps = np.stack([fg_distance_map, bg_distance_map], axis=-1) # (h, w) * 2 -> (h, w, 2) 
        
        image = new_image.transpose((2, 0, 1))[np.newaxis, :, :, :]
        distance_maps = distance_maps.transpose((2, 0, 1))[np.newaxis, :, :, :]

        return image, distance_maps

    def __post_process(self, pred):
        out = cv.resize(pred.squeeze(), self.img_size[::-1]) # (1, 1, h, w)
        out[out > 0.5] = 1
        out[out <= 0.5] = 0
        return out

    def predict(self, image, fg_interactive_map, bg_interactive_map):
        image, distance_maps = self.__pre_process(image, fg_interactive_map, bg_interactive_map)

        self.net.setInput(image, "data")
        self.net.setInput(distance_maps, "iact")

        out = self.net.forward()
        
        # print(out.shape)
        out = self.__post_process(out)

        return out

# prototxt_path = './model/deploy.prototxt'
# caffeModel_path = './model/BRS_DenseNet.caffemodel'

# # net = cv.dnn.readNetFromCaffe(prototxt_path, caffeModel_path)
# # net.setInput(np.random.rand(1, 3, 320, 320), "data")
# # net.setInput(np.random.rand(1, 2, 320, 320), "iact")
# # out = net.forward()
# # print(out.shape)

# image = np.random.rand(300, 500, 3)
# fg_dist_map = np.random.rand(300, 500)
# bg_dist_map = np.random.rand(300, 500)

# net = Net(prototxt_path, caffeModel_path, use_gpu=True, return_spend_time=True)
# out = net.predict(image, fg_dist_map, bg_dist_map)
# print(out.shape)