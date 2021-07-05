import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#  TF_CPP_MIN_LOG_LEVEL = 1 //默认设置，为显示所有信息
#  TF_CPP_MIN_LOG_LEVEL = 2 //只显示error和warining信息
#  TF_CPP_MIN_LOG_LEVEL = 3 //只显示error信息

from our_func_cvpr18 import build
import tensorflow as tf
import numpy as np
from scipy import ndimage
from copy import deepcopy

class IISLD():
    def __init__(self, model_dir="models/ours_cvpr18", threshold=0.5) -> None:
        
        self.threshold = threshold
        
        self.input = tf.placeholder(tf.float32, shape=[None, None, None, 7]) 
        self.output = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.size = tf.placeholder(tf.int32, shape=[2])
        self.network = build(self.input, self.size)

        self.sess=tf.Session()
        saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
        self.sess.run(tf.initialize_all_variables())

        ckpt=tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            # print('loaded '+ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def __process_interactive(self, interactive_map):
        '''
            interactive_map: (h, w), clicked(value=1), unclicked(value=0)
        '''
        if np.any(interactive_map):
            dist_map = ndimage.distance_transform_edt(1 - interactive_map)
            dist_map = np.uint8(np.minimum(np.maximum(dist_map, 0.0), 255.0))
            # print("hello")
        else:
            dist_map = np.full_like(interactive_map, 255, dtype=np.uint8)
        click_map = deepcopy(dist_map)
        click_map[dist_map != 0] = 255
        return dist_map, click_map

    def predict(self, image: np.ndarray, pos_map: np.ndarray, neg_map=None):
        '''
            image: (h, w, 3)
            pos_map: (h, w), clicked(value=1), unclicked(value=0) 
            neg_map: (h, w), clicked(value=1), unclicked(value=0), when no clicked, we can set None. 
        '''
        h, w, _ = image.shape
        pos_dist_map, pos_click_map = self.__process_interactive(pos_map)
        if neg_map is None:
            neg_map = np.zeros((h, w))
        neg_dist_map, neg_click_map = self.__process_interactive(neg_map)

        # import cv2 as cv
        # cv.imshow("fg", pos_dist_map)
        # cv.imshow("bg", neg_dist_map)
        # cv.waitKey(0)

        # image (c=3) + distance map (c=2, pos(1), neg(1)) + click map(c=2, pos(1), neg(1))
        # click map: where clicked will be set 0, unclicked will be set 255.
        input_ = np.expand_dims(np.float32(np.concatenate([image, 
                                                            np.expand_dims(pos_dist_map, axis=2), 
                                                            np.expand_dims(neg_dist_map, axis=2),
                                                            np.expand_dims(pos_click_map, axis=2), 
                                                            np.expand_dims(neg_click_map, axis=2)],axis=2)), axis=0)
        output = self.sess.run([self.network], feed_dict={self.input:input_, self.size:[h, w]})
        output = np.minimum(np.maximum(output, 0.0), 1.0)
        output[output > self.threshold] = 1
        output[output <= self.threshold] = 0
        output = np.uint8(output[0, 0, :, :, 0])

        return output

# from PIL import Image
# import cv2 as cv

# image = np.array(Image.open("./imgs/sample.jpg"))
# fg = np.zeros(image.shape[:2])
# bg = np.zeros(image.shape[:2])
# net = IISLD()

# fg[160, 200] = 1

# out = net.predict(image, fg)

# cv.imshow("out", out*255)
# cv.waitKey(0)

        

