'''
    随机采样相关方法
'''

import os
import random
import numpy as np
import cv2 as cv
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# 0值到1的最小距离
def bwdist(binary_mask):
    distance_map = ndimage.morphology.distance_transform_edt(1 - binary_mask)   
    return distance_map

# 在采点区域随机采一个点
def random_sample_a_points(sample_region:np.ndarray):
    sample_map = np.random.rand(*(sample_region.shape)) * sample_region
    max_value = np.max(sample_map)
    if max_value == 0:
        return None
    else:
        index = np.argmax(sample_map)
        # y = index / ncols , x = index % ncols
        y, x = index // sample_map.shape[1], index % sample_map.shape[1]
    
    return [int(y), int(x)]

# # 在随机采点区域内随机采点
# def sample_points(label:np.ndarray, num_points=10, dmargin=5, dstep=10, d=40):
#     points_list = []
#     label[label == 255] = 0
#     for id in np.unique(label):
#         points = []
#         seg_mask = np.zeros_like(label)
#         seg_mask[label == id] = 1
#         if id == 0:
#             fg = 1 - seg_mask # 前景区域
#             sample_region = np.int8((bwdist(1-seg_mask) < d) - fg) # 背景采点区域
#             # plt.imshow(sample_region)
#         else:
#             sample_region = np.int8(bwdist(1-seg_mask) > dmargin) # 前景采点区域
#         pc = np.zeros_like(seg_mask) # point channel
#         for i in range(num_points):
#             if i > 0:
#                 sample_region = np.int8(sample_region + (bwdist(pc) > dstep) == 2)
#             index = random_sample_a_points(sample_region)
#             if index is None:
#                 break
#             y, x = index
#             pc[y, x] = 1
#             points.append([y, x])
#         points_list.append(points)
#     return points_list

def sample_points_for_region(sample_region, num_points=10, dstep=10):
    points = []
    pc = np.zeros_like(sample_region) # point channel
    for i in range(num_points):
        if i > 0:
            sample_region = np.int8(sample_region + (bwdist(pc) > dstep) == 2)
        index = random_sample_a_points(sample_region)
        if index is None:
            break
        x, y = index
        pc[x, y] = 1
        points.append([x, y])
    return points

def sample_point_by_adaptive_dmargin(binary_mask, num_points=10, dmargin_interval=[0.05, 0.5]):
    points = []
    points_binary_map = np.zeros_like(binary_mask) # positive channel
    max_margin = np.max(bwdist(1-binary_mask))
    for i in range(num_points):
        d_margin = np.random.randint(max_margin * dmargin_interval[0], max_margin * dmargin_interval[1] + 1)
        # print(d_margin)
        sample_region = bwdist(1-binary_mask) > d_margin
        if i == 0:
            sample_map = np.random.rand(*(sample_region.shape)) * sample_region
            max_index = np.argmax(sample_map)
        else:
            max_index = np.argmax(bwdist(points_binary_map) * sample_region)
        # points_binary_map[index // binary_mask.shape[1], index % binary_mask.shape[1]] = 1
        x, y = np.unravel_index(max_index, sample_map.shape)
        points_binary_map[x, y] = 1
        points.append([x, y])
    return points

def sample_point_by_adaptive_dstep(binary_mask, num_points=10, dstep_interval=[0.5, 0.8]):
    points = []
    points_binary_map = np.zeros_like(binary_mask) # positive channel
    sample_region = binary_mask
    max_d_step = np.max(bwdist(1-binary_mask))
    for i in range(num_points):
        d_step = np.random.randint(max_d_step * dstep_interval[0], max_d_step * dstep_interval[1] + 1)
        sample_map = np.random.rand(*(sample_region.shape)) * sample_region
        max_value = np.max(sample_map)
        if max_value == 0:
            break
        else:
            max_index = np.argmax(sample_map)
            # x = index / ncols , y = index % ncols
            # points_binary_map[index // sample_map.shape[1], index % sample_map.shape[1]] = 1
            x, y = np.unravel_index(max_index, sample_map.shape)
            points_binary_map[x, y] = 1
            temp_points_binary_map = np.zeros_like(binary_mask)
            # temp_points_binary_map[index // sample_map.shape[1], index % sample_map.shape[1]] = 1
            temp_points_binary_map[x, y] = 1
            sample_region = np.int8(sample_region + (bwdist(temp_points_binary_map) > d_step) == 2)
            points.append([x, y])
    return points

# 在随机采点区域内随机采点
def sample_points(label:np.ndarray, num_points=10, 
                    dmargin=5, dstep=10, d=40, 
                    dmargin_interval=[0.1, 0.9], dstep_interval=[0.2, 0.2],
                    sample_points_func_name=None):
    if sample_points_func_name == "adaptive_dstep":
        sample_points_func = lambda binary_mask: sample_point_by_adaptive_dstep(binary_mask, num_points, dstep_interval)
    elif sample_points_func_name == "adaptive_dmargin":
        sample_points_func = lambda binary_mask: sample_point_by_adaptive_dmargin(binary_mask, num_points, dmargin_interval)
    else:
        sample_points_func = None
    points_list = []
    label[label == 255] = 0
    # print(np.unique(label))
    for id in np.unique(label):
        if id == 0:
            continue
        else:
            fg = np.zeros_like(label)
            fg[label == id] = 1 # 前景区域
            if sample_points_func is None:
                sample_region = np.int8(bwdist(1 - fg) > dmargin) 
                # cv.imwrite(f"fg_{id}.png", sample_region*255)
                fg_points = sample_points_for_region(sample_region, num_points, dstep) # 前景采点区域
            else:
                fg_points = sample_points_func(fg)
            # bg = 1 - fg # 背景区域
            sample_region = np.int8((bwdist(fg) < d) - fg)
            # cv.imwrite(f"bg_{id}.png", sample_region*255)
            bg_points = sample_points_for_region(sample_region, num_points, dstep) # 背景采点区域
            points_list.append([bg_points, fg_points])
    return points_list

# 随机采极值点
def sample_extreme_points(label, pert=5):
    '''
        参考: https://github.com/scaelles/DEXTR-PyTorch/blob/352ccc76067156ebcf7267b07e0a5e43d32e83d5/dataloaders/helpers.py#L138
    '''
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0])-1)]
        return [int(id_y[sel_id]), int(id_x[sel_id])]
    
    label[label == 255] = 0
    points = []
    for id in np.unique(label):
        if id == 0:
            continue
        iy, ix = np.where(label == id) # row, col
        points.append([find_point(ix, iy, np.where(ix <= (np.min(ix)+pert))), # left
                find_point(ix, iy, np.where(ix >= np.max(ix)-pert)), # right
                find_point(ix, iy, np.where(iy <= np.min(iy)+pert)), # top
                find_point(ix, iy, np.where(iy >= np.max(iy)-pert))]) # bottom
    return points
# def sample_extreme_points(label, pert=5):
#     '''
#         参考: https://github.com/scaelles/DEXTR-PyTorch/blob/352ccc76067156ebcf7267b07e0a5e43d32e83d5/dataloaders/helpers.py#L138
#     '''
#     def find_point(id_x, id_y, ids):
#         sel_id = ids[0][random.randint(0, len(ids[0])-1)]
#         return [int(id_y[sel_id]), int(id_x[sel_id])]
    
#     label[label == 255] = 0
#     points = []
#     for id in np.unique(label):
#         if id == 0:
#             continue
#         iy, ix = np.where(label == id) # row, col
#         points.append([find_point(ix, iy, np.where(ix <= (np.min(ix)+pert))), # left
#                 find_point(ix, iy, np.where(ix >= np.max(ix)-pert)), # right
#                 find_point(ix, iy, np.where(iy <= np.min(iy)+pert)), # top
#                 find_point(ix, iy, np.where(iy >= np.max(iy)-pert))]) # bottom
#     return points

def get_bbox_from_label(binary_label):
    contours, hierarchy = cv.findContours(binary_label, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv.boundingRect(contours[0])
    return [y, x, y + h, x + w]

def sample_bbox_from_label(label, scale=1.0, rand=False):
    def random_value(rand=True):
        if rand:
            return random.gauss(0, 1)
        return 1

    def add_turbulence(bbox, size, v=0.15, rand=True):
        h, w = size
        y_min, x_min, y_max, x_max = bbox[:]
        x_min_new = max(int(x_min - v * random_value(rand) * (x_max - x_min)), 0)
        x_max_new = min(int(x_max + v * random_value(rand) * (x_max - x_min)), w-1)
        y_min_new = max(int(y_min - v * random_value(rand) * (y_max - y_min)), 0)
        y_max_new = min(int(y_max + v * random_value(rand) * (y_max - y_min)), h-1)
        return [y_min_new, x_min_new, y_max_new, x_max_new]
    
    boxes = []
    size = label.shape[:2]
    for i in np.unique(label):
        if i == 0:
            continue
        binary_label = (label==i).astype(np.uint8)
        bbox = get_bbox_from_label(binary_label)
        boxes.append(add_turbulence(bbox, size, (scale-1)/2, rand))
    
    return boxes

# 随机采对象框
def sample_bbox(xml_path, scale=1.0, rand=False):
    '''
        参考: https://github.com/jfzhang95/DeepGrabCut-PyTorch/blob/1e039623b33d1f2ac5b1a586e6f075f8e5a7a70b/dataloaders/utils.py#L64
    '''
    def random_value(rand=True):
        if rand:
            return random.gauss(0, 1)
        return 1

    def add_turbulence(bbox, size, v=0.15, rand=True):
        h, w = size
        y_min, x_min, y_max, x_max = bbox[:]
        x_min_new = max(int(x_min - v * random_value(rand) * (x_max - x_min)), 0)
        x_max_new = min(int(x_max + v * random_value(rand) * (x_max - x_min)), w-1)
        y_min_new = max(int(y_min - v * random_value(rand) * (y_max - y_min)), 0)
        y_max_new = min(int(y_max + v * random_value(rand) * (y_max - y_min)), h-1)
        return [y_min_new, x_min_new, y_max_new, x_max_new]

    boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w, h = int(size.find('width').text), int(size.find('height').text)
    for o in root.iter('object'):
        box = o.find('bndbox')
        # [ymin, xmin, ymax, xmax]
        bbox = [int(box.find('ymin').text), int(box.find('xmin').text), 
                int(box.find('ymax').text), int(box.find('xmax').text)]
        boxes.append(add_turbulence(bbox, (h, w), (scale-1)/2, rand))
    return boxes

def show_infos(label, points=None, extreme_points=None, bboxes=None):
    
    infos = [points, extreme_points, bboxes]
    assert any(infos), "all infos is None."

    nums_object = len(bboxes) + 1
    color = [ int(255 / nums_object * (i + 1)) for i in range(nums_object)]
    label = label * 10

    num = 1
    for i in infos:
        if i is not None:
            num += 1

    plt.figure(figsize=(20, 8))
    plt.subplot(1, num, 1)
    plt.title("label")
    plt.imshow(label)
    index = 2
    if points is not None:
        plt.subplot(1, num, index)
        plt.title("points")
        temp = label.copy()
        for i, point in enumerate(points):
            for p in point:
                cv.circle(temp, tuple(p[::-1]), 3, (color[i]), -1) # point(x, y)
        plt.imshow(temp)
        index += 1

    if extreme_points is not None:
        plt.subplot(1, num, index)
        plt.title("extreme points")
        temp = label.copy()
        for i, point in enumerate(extreme_points):
            for p in point:
                cv.circle(temp, tuple(p[::-1]), 3, (color[i]), -1) # point(x, y)
        plt.imshow(temp)
        index += 1

    if points is not None:
        plt.subplot(1, num, index)
        temp = label.copy()
        plt.title("bboxes")
        for i, point in enumerate(bboxes):
            cv.rectangle(temp, (point[1], point[0]), (point[3], point[2]), (color[i]), 2) # point(x, y)
        plt.imshow(temp)
    plt.show()

def random_sample(dir, file_name, plot=False):
    xml_path = os.path.join(dir, f"{file_name}.xml")
    label_path = os.path.join(dir, f"{file_name}.png")

    label = np.array(Image.open(label_path))
    points = sample_points(label)
    extreme_points = sample_extreme_points(label)
    bboxes = sample_bbox(xml_path)
    
    if plot:
        show_infos(label, points, extreme_points, bboxes)
    
    return points, extreme_points, bboxes


def points_list_to_binary_image(size, points_list, combine=True, stack_axis=-1):
    binary_list = []
    for points in points_list:
        binary = points_to_binary_image(size, points)
        binary_list.append(binary)
    if combine:
        binary_list = np.stack(binary_list, axis=stack_axis)
    
    return binary_list

# 将点转为二值图
def points_to_binary_image(size, points):
    binary = np.zeros(size, dtype=np.uint8)
    for p in points:
        y, x =  p
        binary[y, x] = 1
    return binary

def bbox_to_binary_image(size, bbox):
    binary = np.zeros(size, dtype=np.uint8)
    miny, minx, maxy, maxx = bbox
    binary[miny, minx:maxx] = 1
    binary[maxy, minx:maxx] = 1
    binary[miny:maxy, minx] = 1
    binary[miny:maxy, maxx] = 1
    return binary