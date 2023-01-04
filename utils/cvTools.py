import cv2
import matplotlib.pyplot as plt

def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def img_resize(img, hight):
    (h, w) = img.shape[0], img.shape[1]
    r = h / hight
    width = w / r
    img = cv2.resize(img, (int(width), int(hight)))
    return img


def cut_card(img):
    h = img.shape[0]
    return img[h//2:h//3*2]


def plot_images(row, col, imgs, titles=None):
    i = 1
    for img in imgs:
        plt.subplot(row, col, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles is not None:
            plt.title(titles[i-1])
        plt.xticks([])
        plt.yticks([])
        i += 1
    plt.show()


def splitByRect(img, rects):
    blocks = []
    for (x, y, w, h) in rects:
        blocks.append(img[y:y+h, x:x+w])

    return blocks


def drawNumbersBack(img, numbers, points, blockRects, fontsize):
    cur_img = img.copy()
    high = cur_img.shape[0]
    i = 0
    for number, point in zip(numbers, points):
        x, y, w, h = point
        cur_img = cv2.putText(cur_img, number, (x + blockRects[i//4][0], y+high//2), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 255), 2)
        i += 1
    return cur_img