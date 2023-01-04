import cv2
import numpy as np


def sobelxGrad(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    minval, maxval = np.min(sobelx), np.max(sobelx)
    sobelx = (255 * ((sobelx - minval) / (maxval - minval)))
    sobelx = sobelx.astype('uint8')
    return sobelx