import os
import cv2 as cv
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from net import FCANet

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

net = FCANet()

# ----- dataset dir -----
voc_root_dir = '/home/guiyan/workspaces/datasets/voc2012/VOCdevkit/VOC2012'
mask_dir = os.path.join(voc_root_dir, 'SegmentationObject')
image_dir = os.path.join(voc_root_dir, 'JPEGImages')
splits_file = os.path.join(voc_root_dir, 'ImageSets/Segmentation/val.txt')
interactives_dir = "/home/guiyan/workspaces/bingqiangzhou/iis/2020/interactives"

# ----- get all file name from val dataset -----
with open(os.path.join(splits_file), "r") as f:
    file_names = [x.strip() for x in f.readlines()]

# ----- output dir -----

out_dir = './seg_result'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

excel_path = os.path.join(out_dir, 'out.xlsx')
writer = pd.ExcelWriter(excel_path)

max_num_points = 20
for j in range(max_num_points):
    result_dir = os.path.join(out_dir, f"points_{j+1}")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    data_list = []
    for name in tqdm(file_names, desc=f"points-{j+1}"):
        image_np = np.array(Image.open(os.path.join(image_dir, name+'.jpg')))
        label_np = np.array(Image.open(os.path.join(mask_dir, name+'.png')))
        label_np[label_np == 255] = 0
        
        ious = []
        f1_scores = []
        spend_times = []
        ids = np.unique(label_np)
        nums_object = len(ids) - 1
        for i in ids:
            if i == 0:
                continue

            gt = np.uint8(label_np == i)
            fg_interactive_map = np.array(Image.open(f"{interactives_dir}/{name}_fg_interactivemap_object{i}_point{j+1}.png"))
            bg_interactive_map = np.array(Image.open(f"{interactives_dir}/{name}_bg_interactivemap_object{i}_point{j+1}.png"))
            first_click_interactive_map = np.array(Image.open(f"{interactives_dir}/{name}_bg_interactivemap_object{i}_point1.png"))
            
            fg_interactive_map[fg_interactive_map == 255] = 1
            bg_interactive_map[bg_interactive_map == 255] = 1
            first_click_interactive_map[first_click_interactive_map == 255] = 1

            outputs, spend_time = net.predict(image_np, fg_interactive_map, bg_interactive_map, first_click_interactive_map)
            
            iou_v = iou(outputs, gt)
            ious.append(iou_v)
            f1_score_v = f1_score(outputs, gt)
            f1_scores.append(f1_score_v)
            spend_times.append(spend_time)

            cv.imwrite(os.path.join(result_dir, f"{name}_object{i}_point{j+1}_iou{iou_v}_f1score{f1_score_v}.png"), outputs*255)
        
        mean_iou = np.mean(ious)
        mean_f1_score = np.mean(f1_scores)
        mean_spend_times = np.mean(spend_times)
        for k in range(len(ious)):
            data_list.append([name, nums_object, k, ious[k], mean_iou, f1_scores[k], mean_f1_score, spend_times[k], mean_spend_times])
            # print([name, nums_object, k+1, ious[k], mean_iou, f1_scores[k], mean_f1_score])
        
        # break
    data_list = pd.DataFrame(data_list, #index=file_names, 
                                columns=["image name" , "num objects", "object index",  
                                "iou", "mean iou", "f1-score", "mean f1-score", 'speed time', 'mean speed time'])
    data_list.to_excel(writer, sheet_name=f"point_{j+1}",  float_format="%.6f", index=False)
    writer.save()
writer.close()