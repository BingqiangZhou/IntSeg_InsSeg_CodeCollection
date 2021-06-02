import os
import cv2 as cv
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from net import RefineMask

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
    row_index, col_index = linear_sum_assignment(-iou_matrix)
    return row_index, col_index

# net = RefineMask(threshold=0.8)
net = RefineMask() # threshold=0.5
net_name = 'RefineMask'

# ----- dataset dir -----
voc_root_dir = r'E:\Datasets\iis_datasets\VOCdevkit\VOC2012'
mask_dir = os.path.join(voc_root_dir, 'SegmentationObject')
image_dir = os.path.join(voc_root_dir, 'JPEGImages')
splits_file = os.path.join(voc_root_dir, 'ImageSets/Segmentation/val.txt')

# ----- get all file name from val dataset -----
with open(os.path.join(splits_file), "r") as f:
    file_names = [x.strip() for x in f.readlines()]

# ----- output dir -----
results_dir = r'E:\Workspaces\zbq\projects\IS\seg_results'
out_dir =os.path.join(results_dir, net_name)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

excel_path = os.path.join(results_dir, net_name +'.xlsx')
writer = pd.ExcelWriter(excel_path)

data_list = []

for name in tqdm(file_names, desc=f"{net_name}"):
    image_np = np.array(Image.open(os.path.join(image_dir, name+'.jpg')))
    label_np = np.array(Image.open(os.path.join(mask_dir, name+'.png')))
    label_np[label_np == 255] = 0
    

    ids = np.unique(label_np)
    nums_object = len(ids) - 1
    outputs, _, _, inference_time = net.predict(image_np) 
    mean_inference_time_for_each_object = inference_time / nums_object
    nums_pred_object = len(outputs)  # 后续处理预测对象数小于label中对象数的情况

    # label中对象的mask 与 预测的mask 一一求交叠率，再通过匈牙利匹配算法求得“完美匹配”
    # 构建iou矩阵
    iou_matrix = np.zeros((nums_object, nums_pred_object)) 
    for i in range(nums_object):
        for j in range(nums_pred_object):
            gt = np.uint8(label_np == (i+1))
            iou_matrix[i, j] = iou(outputs[j], gt)
    # print(iou_matrix)
    # row_index, col_index
    gt_index, pred_mask_index = max_iou_assignment(iou_matrix) # 得到“完美配置”（最大匹配）
    # 可视化 对应配置结果
    # for i in range(nums_object):
    #     plt.subplot(nums_object, 2, 2*i + 1)
    #     gt = np.uint8(label_np == (i+1))
    #     plt.imshow(gt)
    #     plt.subplot(nums_object, 2, 2*i + 2)
    #     plt.imshow(outputs[index[i]])
    # plt.show()

    iou_list = []
    f1_score_list = []

    bg = np.ones_like(label_np) # 记录预测的背景图像
    predict_label = np.zeros_like(label_np)
    # 保存相关指标数据
    for i in range(nums_object):
        gt = np.uint8(label_np == (i+1))

        if i in gt_index:
            iou_v = 0
            f1_score_v = 0
        else:
            iou_v = iou_matrix[i][pred_mask_index[i]]
            f1_score_v = f1_score(outputs[pred_mask_index[i]], gt)

            bg[outputs[pred_mask_index[i]] > 0] = 0 
            predict_label[outputs[pred_mask_index[i]] > 0] = i + 1
            
            cv.imwrite(os.path.join(out_dir, f"{name}_object{i}_iou{iou_v}_f1score{f1_score_v}.png"), outputs[pred_mask_index[i]]*255)
        
        iou_list.append(iou_v)
        f1_score_list.append(f1_score_v)

    plt.imsave(os.path.join(out_dir, f"{name}_predict.png"), predict_label)
    
    # 计算背景图像相关指标
    bg_gt = np.uint8(label_np == 0)
    bg_iou = iou(bg, bg_gt)
    bg_f1_score = f1_score(bg, bg_gt)

    mean_iou = np.mean(iou_list)
    mean_f1_score = np.mean(f1_score_list)
    mean_iou_with_bg = (mean_iou * nums_object + bg_iou) / (nums_object + 1)
    mean_f1_score_with_bg = (mean_f1_score * nums_object + bg_f1_score) / (nums_object + 1)
    for k in range(len(iou_list)):
        data_list.append([name, nums_object, k, iou_list[k], mean_iou, f1_score_list[k], mean_f1_score, 
                        mean_inference_time_for_each_object, bg_iou, mean_iou_with_bg, bg_f1_score, mean_f1_score_with_bg])
    #     print([name, nums_object, k, iou_list[k], mean_iou, f1_score_list[k], mean_f1_score, 
    #                     mean_inference_time_for_each_object, bg_iou, mean_iou_with_bg, bg_f1_score, mean_f1_score_with_bg])
    # break

data_list = pd.DataFrame(data_list, #index=file_names, 
                            columns=["image name" , "num objects", "object index",  "iou", "mean iou", "f1-score", "mean f1-score",
                            "mean inference time for each object", "bg iou", "mean iou with bg", "bg f1-score", "mean f1-score with bg"])
data_list.to_excel(writer, float_format="%.6f", index=False)
writer.close()
    


