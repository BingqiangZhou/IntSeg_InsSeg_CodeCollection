import os
from PIL import Image
import numpy as np
import cv2 as cv
import tkinter as tk
import tkinter.filedialog as filedialog

from net import DeepGrabCut

net = DeepGrabCut()

start_drawing = False
drawing_finished = False
bbox = [1e6, 1e6, 0, 0]
last_x, last_y = 0, 0
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

def update_bbox(bbox, x, y, size):
    h, w = size 
    bbox[0] = max(min(bbox[0], x), 0)
    bbox[1] = max(min(bbox[1], y), 0)
    bbox[2] = min(max(bbox[2], x), w-1)
    bbox[3] = min(max(bbox[3], y), h-1)
    return bbox

def interactives(event, x, y, flags, param):
    global start_drawing, drawing_finished, show_image, image, bbox, last_x, last_y
    if event == cv.EVENT_LBUTTONDOWN: # left button down: begin interactives
        start_drawing = True
        drawing_finished = False
        last_x, last_y = x, y
        bbox = update_bbox(bbox, x, y, show_image.shape[:2])
    elif event == cv.EVENT_MOUSEMOVE:
        if start_drawing and not drawing_finished:
            cv.line(show_image, (last_x, last_y), (x, y), (255, 0, 0), 2)
            last_x, last_y = x, y
            bbox = update_bbox(bbox, x, y, show_image.shape[:2])
    elif event == cv.EVENT_LBUTTONUP: # left button up: end interactive
        cv.line(show_image, (last_x, last_y), (x, y), (255, 0, 0), 2)
        bbox = update_bbox(bbox, x, y, show_image.shape[:2])
        drawing_finished = True
    elif event == cv.EVENT_RBUTTONUP: # right button up: cancel interactive
        start_drawing = False
        drawing_finished = False
        show_image = image.copy()
        cv.imshow("image", show_image[:, :, ::-1])

image_path = get_open_file_path()
image = np.array(Image.open(image_path))

cv.imshow("image", image[:, :, ::-1])
cv.setMouseCallback("image", interactives)
show_image = image.copy()

while True:
    cv.imshow("image", show_image[:, :, ::-1])
    key = cv.waitKey(100)
    if key == 27:
        break
    if cv.getWindowProperty("image", cv.WND_PROP_VISIBLE) <= 0: # window has been closed.
        break

    if key == 112 or key == 80: # "p" or "P"
        if drawing_finished:
            # segment
            output = net.predict(image, bbox)
            cv.imshow("out", output*255)

            # save predict result and interactives
            image_name, image_ext = os.path.splitext(image_path)
            save_path_of_image_with_interactives = f"{image_name}-interactives{image_ext}"
            save_path_of_predict_result = f"{image_name}-result.png"
            cv.imwrite(save_path_of_image_with_interactives, show_image[:, :, ::-1])
            cv.imwrite(save_path_of_predict_result, output*255)
            # print(bbox)
            print(f"predict result save in {save_path_of_predict_result}")
            
            # wait "out" window to close
            wait_cv_window("out")
            
            # init
            show_image = image.copy()
    elif key == 111 or key == 79: # "o" or "O"
        # open a new image to segment
        image_path = get_open_file_path()
        image = np.array(Image.open(image_path))
        
        # init
        show_image = image.copy()

root.destroy()
