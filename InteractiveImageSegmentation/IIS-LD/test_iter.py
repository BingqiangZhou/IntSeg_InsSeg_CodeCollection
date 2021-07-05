import os
from PIL import Image
import numpy as np
import cv2 as cv
import tkinter as tk
import tkinter.filedialog as filedialog

from net import IISLD

point_list = [] # [[x, y, 0/1], ...] # 0 is bg and 1 is fg
last_result = None
root = tk.Tk()
root.withdraw() # hide the main window

def get_open_file_path():
    image_path = filedialog.askopenfilename(initialdir='./imgs', 
                                    filetypes=[("JPG", ".jpg"), ("PNG", ".png"), ("GIF", ".gif"), 
                                                ("TIF", ".tif"), ("bmp", ".bmp"), ("ALL TYPE", ".*")])
    return image_path

def wait_cv_window(window_name):
    while cv.waitKey(100) != 27: # loop if not get ESC
        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) <= 0: # window has been closed.
            break
    cv.destroyWindow(window_name)

def points_to_trimap(point_list, size):
    fg = np.zeros(size)
    bg = np.zeros(size)
    for point in point_list:
        x, y, index = point
        if index == 1:
            fg[y, x] = 1
        else:
            bg[y, x] = 1
    
    return fg, bg

def combine_result_and_interactives(result, interactives_image, alpha=0.6):
    show_image = cv.addWeighted(interactives_image, alpha, result[:, :, None] * [255, 0, 0], 1-alpha, 0, dtype=cv.CV_32F)
    return show_image


def click(event, x, y, flags, param):
    global point_list, interactives_image, image, last_result
    num_points = len(point_list)
    if flags == cv.EVENT_FLAG_CTRLKEY and event == cv.EVENT_LBUTTONUP: # cancel last click: press 'ctrl' key and left button
        if len(point_list) > 0:
            point_list.pop()
            interactives_image = image.copy()
            for point in point_list:
                x, y, index = point
                color = (255, 0, 0) if index == 1 else (0, 255, 0)
                cv.circle(interactives_image, (x, y), 3, color, -1)
    elif event == cv.EVENT_LBUTTONUP: # left button: fg
        point_list.append([x, y, 1])
        cv.circle(interactives_image, (x, y), 3, (255, 0, 0), -1)
    elif event == cv.EVENT_RBUTTONUP: # right button: bg
        point_list.append([x, y, 0])
        cv.circle(interactives_image, (x, y), 3, (0, 255, 0), -1)
    
    if len(point_list) != num_points:
        fg, bg = points_to_trimap(point_list, interactives_image.shape[:2])
        output = net.predict(image, fg, bg)
        last_result = output.copy()
        show_image = combine_result_and_interactives(output, interactives_image).astype(np.uint8)
        cv.imshow("image", show_image[:, :, ::-1])

image_path = get_open_file_path()
image = np.array(Image.open(image_path))
net = IISLD()
cv.imshow("image", image[:, :, ::-1])
cv.setMouseCallback("image", click)
interactives_image = image.copy()
show_image = image.copy()

while True:
    key = cv.waitKey(100)
    if key == 27:
        break
    if cv.getWindowProperty("image", cv.WND_PROP_VISIBLE) <= 0: # window has been closed.
        break

    if key == 111 or key == 79: # "o" or "O"
        # open a new image to segment
        image_path = get_open_file_path()
        image = np.array(Image.open(image_path))
        
        # init
        show_image = image.copy()
        cv.imshow("image", show_image[:, :, ::-1])
    elif key == 67 or key == 99: # "c" or "C"
        show_image = image.copy()
        cv.imshow("image", show_image[:, :, ::-1])
    elif key == 115 or key == 83: # "s" or "S"
        if last_result is not None:
            # save predict result and interactives
            image_name, image_ext = os.path.splitext(image_path)
            save_path_of_image_with_interactives = f"{image_name}-iter-interactives{image_ext}"
            save_path_of_predict_result = f"{image_name}-iter-result.png"
            cv.imwrite(save_path_of_image_with_interactives, show_image[:, :, ::-1])
            cv.imwrite(save_path_of_predict_result, last_result*255)
            print(f"predict result save in {save_path_of_predict_result}")

root.destroy()
