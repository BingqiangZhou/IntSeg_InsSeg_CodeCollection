from PIL import Image
import numpy as np

import cv2 as cv

from net import DEXTR

click_times = 0

def click(event, x, y, flags, param):
    global click_times, show_image, image, extreme_points
    if event == cv.EVENT_LBUTTONUP:
        click_times += 1
        extreme_points.append([x, y])
        print(f"{click_times}-th clicked, {[x, y]}")
        cv.circle(show_image, (x, y), 3, (255, 0, 0), -1)
        cv.imshow("image", show_image[:, :, ::-1])
    elif event == cv.EVENT_RBUTTONUP:
        x, y = extreme_points.pop()
        print(f"remove {click_times}-th clicked, {[x, y]}")
        click_times -= 1
        show_image = image.copy()
        for point in extreme_points:
            x, y = point
            cv.circle(show_image, (x, y), 3, (255, 0, 0), -1)
        cv.imshow("image", show_image[:, :, ::-1])

image = np.array(Image.open("./ims/bear.jpg"))
extreme_points = []

net = DEXTR()

cv.imshow("image", image[:, :, ::-1])
cv.setMouseCallback("image", click)
show_image = image.copy()

while True:
    if click_times == 4:
        extreme_points_np = np.array(extreme_points)
        output = net.predict(image, extreme_points_np)
        cv.imshow("out", output*255)
        cv.waitKey(0)

        click_times = 0
        show_image = image.copy()
        cv.imshow("image", show_image[:, :, ::-1])
        extreme_points = []
    
    cv.imshow("image", show_image[:, :, ::-1])

    key = cv.waitKey(1)
    if key == 27:
        break
