import cv2

def topHat(img, k1=9, k2=3):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k1, k2))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, rectKernel)
    return tophat


def morphClose(img, k1=9, k2=3,n=10):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k1, k2))
    dilate = cv2.dilate(img, rectKernel, n)
    erosion = cv2.erode(dilate, rectKernel, n)
    return erosion

