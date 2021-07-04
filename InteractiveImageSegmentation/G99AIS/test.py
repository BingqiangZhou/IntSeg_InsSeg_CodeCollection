import os
from PIL import Image
import numpy as np
import cv2 as cv
import tkinter as tk
import tkinter.filedialog as filedialog

from net import G99AIS

point_list = [] # [[x, y, 0/1], ...] # 0 is bg and 1 is fg
root = tk.Tk()
root.withdraw() # hide the main window

def get_open_file_path():
    image_path = filedialog.askopenfilename(initialdir='./ims', 
                                    filetypes=[("JPG", ".jpg"), ("PNG", ".png"), ("GIF", ".gif"), 
                                                ("TIF", ".tif"), ("bmp", ".bmp"), ("ALL TYPE", ".*")])
    return image_path

def wait_cv_window(window_name):
    while cv.waitKey(100) != 27: # loop if not get ESC
        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) <= 0: # window has been closed.
            break
    cv.destroyWindow(window_name)

def points_to_trimap(point_list, size):
    h, w = size
    trimap = np.zeros((h, w, 2))
    for point in point_list:
        x, y, index = point
        trimap[y, x, index] = 1
    
    return trimap


def click(event, x, y, flags, param):
    global point_list, show_image, image
    if flags == cv.EVENT_FLAG_CTRLKEY and event == cv.EVENT_LBUTTONUP: # cancel last click: press 'ctrl' key and left button
        if len(point_list) > 0:
            point_list.pop()
            show_image = image.copy()
            for point in point_list:
                x, y, index = point
                color = (255, 0, 0) if index == 1 else (0, 255, 0)
                cv.circle(show_image, (x, y), 3, color, -1)
            cv.imshow("image", show_image[:, :, ::-1])
    elif event == cv.EVENT_LBUTTONUP: # left button: fg
        point_list.append([x, y, 1])
        cv.circle(show_image, (x, y), 3, (255, 0, 0), -1)
        cv.imshow("image", show_image[:, :, ::-1])
    elif event == cv.EVENT_RBUTTONUP: # right button: bg
        point_list.append([x, y, 0])
        cv.circle(show_image, (x, y), 3, (0, 255, 0), -1)
        cv.imshow("image", show_image[:, :, ::-1])

image_path = get_open_file_path()
image = np.array(Image.open(image_path))

net = G99AIS()

cv.imshow("image", image[:, :, ::-1])
cv.setMouseCallback("image", click)
show_image = image.copy()

while True:
    cv.imshow("image", show_image[:, :, ::-1])
    key = cv.waitKey(100)
    if key == 27:
        break
    if cv.getWindowProperty("image", cv.WND_PROP_VISIBLE) <= 0: # window has been closed.
        break

    if key == 112 or key == 80: # "p" or "P"
        # segment
        if len(point_list) > 0:
            trimap = points_to_trimap(point_list, show_image.shape[:2])
            output = net.predict(image, trimap)
            cv.imshow("out", output*255)

            # save predict result and interactives
            image_name, image_ext = os.path.splitext(image_path)
            save_path_of_image_with_interactives = f"{image_name}-interactives{image_ext}"
            save_path_of_predict_result = f"{image_name}-result.png"
            cv.imwrite(save_path_of_image_with_interactives, show_image[:, :, ::-1])
            cv.imwrite(save_path_of_predict_result, output*255)
            print(f"predict result save in {save_path_of_predict_result}")
            
            # wait "out" window to close
            wait_cv_window("out")
            
            # init
            click_times = 0
            show_image = image.copy()
            cv.imshow("image", show_image[:, :, ::-1])
            extreme_points = []
    elif key == 111 or key == 79: # "o" or "O"
        # open a new image to segment
        image_path = get_open_file_path()
        image = np.array(Image.open(image_path))
        
        # init
        click_times = 0
        show_image = image.copy()
        cv.imshow("image", show_image[:, :, ::-1])
        extreme_points = []

root.destroy()
