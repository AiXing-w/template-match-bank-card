import cv2


def getRect(img, minVal=300):
    contour, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > minVal:
            rects.append([x, y, w, h])

    rects = sorted(rects, key=lambda x:x[0])
    return rects
