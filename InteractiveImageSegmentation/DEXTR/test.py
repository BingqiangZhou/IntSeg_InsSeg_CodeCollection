import os
from PIL import Image
import numpy as np
import cv2 as cv
import tkinter as tk
import tkinter.filedialog as filedialog

from net import DEXTR

click_times = 0
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

def click(event, x, y, flags, param):
    global click_times, show_image, image, extreme_points
    if event == cv.EVENT_LBUTTONUP: # left button: interactives
        click_times += 1
        extreme_points.append([x, y])
        print(f"{click_times}-th clicked, {[x, y]}")
        cv.circle(show_image, (x, y), 3, (255, 0, 0), -1)
        cv.imshow("image", show_image[:, :, ::-1])
    elif event == cv.EVENT_RBUTTONUP: # right button: cancel last interactive
        if len(extreme_points) > 0:
            x, y = extreme_points.pop()
            print(f"remove {click_times}-th clicked, {[x, y]}")
            click_times -= 1
            show_image = image.copy()
            for point in extreme_points:
                x, y = point
                cv.circle(show_image, (x, y), 3, (255, 0, 0), -1)
            cv.imshow("image", show_image[:, :, ::-1])

image_path = get_open_file_path()
image = np.array(Image.open(image_path))
extreme_points = []

net = DEXTR()

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
        if click_times == 4:
            # segment
            extreme_points_np = np.array(extreme_points)
            output = net.predict(image, extreme_points_np)
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
            extreme_points = []
    elif key == 111 or key == 79: # "o" or "O"
        # open a new image to segment
        image_path = get_open_file_path()
        image = np.array(Image.open(image_path))
        
        # init
        click_times = 0
        show_image = image.copy()
        extreme_points = []

root.destroy()
