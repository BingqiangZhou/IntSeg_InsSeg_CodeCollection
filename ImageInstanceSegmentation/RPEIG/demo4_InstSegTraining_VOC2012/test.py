import os
import scipy.io as sio
import numpy as np
from PIL import Image
import cv2 as cv
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering # https://scikit-learn.org/stable/modules/clustering.html#
from scipy.optimize import linear_sum_assignment

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def iou(binary_predict, binary_target, epsilon=1e-6):
    add_result = binary_predict + binary_target
    union = (add_result == 2).astype(np.uint8)
    intersection = add_result - union
    
    return np.sum(union) / (np.sum(intersection) + epsilon)

def f1_score(binary_predict, binary_target,  epsilon=1e-6):
    add_result = binary_predict + binary_target
    union = (add_result == 2).astype(np.uint8)
    precision = np.sum(union) /  (np.sum(binary_predict) + epsilon)
    recall = np.sum(union) /  (np.sum(binary_target) + epsilon)

    return 2 * (precision * recall) / (precision + recall + epsilon)

# reference: 
#   https://blog.csdn.net/qq_25602729/article/details/108377648
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
def max_iou_assignment(iou_matrix):
    indx_1, indx_2 = linear_sum_assignment(-iou_matrix)
    return indx_1, indx_2

# 获取图像名列表
voc_root_dir = r'E:\Datasets\iis_datasets\VOCdevkit\VOC2012'
label_dir = os.path.join(voc_root_dir, 'SegmentationObject')
image_dir = os.path.join(voc_root_dir, 'JPEGImages')
splits_file = os.path.join(voc_root_dir, 'ImageSets/Segmentation/val.txt')

use_time_file = "./useTime.txt"
fea_map_dir = r'E:\Workspaces\zbq\projects\IS\RPEIG\demo4_InstSegTraining_VOC2012\feature_maps'

# ----- get the time of inference -----
with open(os.path.join(use_time_file), "r") as f:
    use_time_list = [float(x.strip()) for x in f.readlines()]
# print(use_time_list)

# ----- get all file name from val dataset -----
with open(os.path.join(splits_file), "r") as f:
    file_names = [x.strip() for x in f.readlines()]

# ----- output dir -----
results_dir = r'E:\Workspaces\zbq\projects\IS\seg_results'
out_dir =os.path.join(results_dir, 'RPEIG')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


excel_path = os.path.join(results_dir, 'RPEIG.xlsx')
writer = pd.ExcelWriter(excel_path)

data_list = []

for k, image_name in enumerate(tqdm(file_names, desc="RPEIG")):
    # 根据图像名获取到原图像以及label
    image_path = os.path.join(image_dir, image_name + '.jpg')
    label_path = os.path.join(label_dir, image_name + '.png')
    image = np.array(Image.open(image_path))
    label = np.array(Image.open(label_path))
    label[label == 255] = 0
    num_objects = len(np.unique(label)) # 包括背景

    image_h, image_w = image.shape[:2] # h, w of  the source image

    # 根据图像名获取特征图
    data = sio.loadmat(os.path.join(fea_map_dir, image_name + '.mat'))
    fea_map = data['feaMap']

    # 特征图聚类得到预测的label，簇的数量由真实的label中对象数确定
    fea_map_h,  fea_map_w, fea_map_channel = fea_map.shape
    fea_map_reshaped = np.reshape(fea_map, (fea_map_h * fea_map_w, fea_map_channel))
    # kmeans = KMeans(4).fit(fea_map_reshaped)
    # labels = kmeans.labels_.reshape(h, w)
    sc = SpectralClustering(num_objects).fit(fea_map_reshaped)
    predict_labels = sc.labels_.reshape(fea_map_h, fea_map_w)
    # print(predict_labels.shape, np.unique(predict_labels))
    # plt.imshow(predict_labels)
    # plt.show()

    # 上采样到原图大小
    predict_labels = cv.resize(predict_labels.astype(np.uint8), (image_w, image_h), interpolation=cv.INTER_NEAREST)
    # print(predict_labels.shape, np.unique(predict_labels))
    # plt.imshow(predict_labels)
    # plt.show()

    # 计算预测的mask与真实的mask做匈牙利匹配，得到最大匹配 “完美匹配”
    # 构建iou矩阵
    iou_matrix = np.zeros((num_objects, num_objects)) 
    for i in range(num_objects):
        for j in range(num_objects):
            gt = np.uint8(label == (i))
            predict_mask = np.uint8(predict_labels == j)
            iou_matrix[i, j] = iou(predict_mask, gt)
    # print(iou_matrix)
    _, index = max_iou_assignment(iou_matrix) # 得到“完美配置”（最大匹配）

    # 可视化 对应匹配结果
    # for i in range(num_objects):
    #     plt.subplot(num_objects, 2, 2*i + 1)
    #     gt = np.uint8(label == (i))
    #     plt.imshow(gt)
    #     plt.subplot(num_objects, 2, 2*i + 2)
    #     predict_mask = np.uint8(predict_labels == index[i])
    #     plt.imshow(predict_mask)
    # plt.show()

    # 保存mask，计算相关指标，并保存到excel中
    iou_list = []
    f1_score_list = []
    new_predict_label = np.zeros_like(predict_labels) # 聚类后的label的值没有规律，比如可能背景值不是0，而是其他的，这里规范一下
    for i in range(num_objects):
        gt = np.uint8(label == i)
        predict_mask = np.uint8(predict_labels == index[i])

        iou_v = iou_matrix[i][index[i]]
        f1_score_v = f1_score(predict_mask, gt)
        
        iou_list.append(iou_v)
        f1_score_list.append(f1_score_v)

        if i != 0: # 不保留背景mask
            new_predict_label[predict_labels == index[i]] = i
            cv.imwrite(os.path.join(out_dir, f"{image_name}_object{i}_iou{iou_v}_f1score{f1_score_v}.png"), predict_mask*255)
    plt.imsave(os.path.join(out_dir, f"{image_name}_predict.png"), new_predict_label)

    mean_iou = np.mean(iou_list[1:])
    mean_f1_score = np.mean(f1_score_list[1:])
    mean_iou_with_bg = np.mean(iou_list)
    mean_f1_score_with_bg = np.mean(f1_score_list)
    mean_inference_time_for_each_object = use_time_list[k] / (num_objects - 1)
    for k in range(1, len(iou_list)):
        data_list.append([image_name, num_objects-1, k-1, iou_list[k], mean_iou, f1_score_list[k], mean_f1_score, 
                        mean_inference_time_for_each_object, iou_list[0], mean_iou_with_bg, f1_score_list[0], mean_f1_score_with_bg])
        # print([image_name, num_objects, k, iou_list[k], mean_iou, f1_score_list[k], mean_f1_score, 
        #                 mean_inference_time_for_each_object, iou_list[0], mean_iou_with_bg, f1_score_list[0], mean_f1_score_with_bg])
    # break

data_list = pd.DataFrame(data_list, #index=file_names, 
                            columns=["image name" , "num objects", "object index",  "iou", "mean iou", "f1-score", "mean f1-score",
                            "mean inference time for each object", "bg iou", "mean iou with bg", "bg f1-score", "mean f1-score with bg"])
data_list.to_excel(writer, float_format="%.6f", index=False)
writer.close()

