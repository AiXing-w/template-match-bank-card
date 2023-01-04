import cv2
import numpy as np


def templateMatch(img, template):
    cur_img = cv2.resize(img, (100, 150))
    scores = []
    for i in range(10):
        result = cv2.matchTemplate(cur_img, template[i], cv2.TM_CCOEFF)
        (_, score, _, _) = cv2.minMaxLoc(result)

        scores.append(score)
    return str(np.argmax(scores))