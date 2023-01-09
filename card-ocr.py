import cv2
import argparse
import os

from utils.cvTools import cv_show, img_resize, cut_card, splitByRect, plot_images, drawNumbersBack
from utils.dilateAndErosion import topHat, morphClose
from utils.gradX import sobelxGrad
from utils.contours import getRect
from utils.matchTemplate import templateMatch


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input image", default='./images/credit_card_01.png')
ap.add_argument("-t", "--template", help="path to template OCR-A image", default='./cuted_template')
ap.add_argument("-high", '--chigh', help="the high of the card", default=200)  # 这里不建议更改
ap.add_argument("-s", "--imgshow", help="option for image show", default=True)
args = vars(ap.parse_args())


def getCardNumbersBlocks(card):# 对银行卡进行处理
    cuted_card = cut_card(card)  # 裁剪
    gray = cv2.cvtColor(cuted_card, cv2.COLOR_BGR2GRAY)  # 灰度化
    tophat = topHat(gray)  # 顶帽
    gradx = sobelxGrad(tophat)  # sobel x方向梯度
    close = morphClose(gradx)  # 闭操作
    ret, thresh = cv2.threshold(close, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)  # 二值化
    close = morphClose(thresh, k1=5, k2=5)  # 闭操作
    rects = getRect(close)  # 获取矩形框
    blocks = splitByRect(cv2.threshold(gray, 0,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1], rects)  # 按照矩形框划分图像
    return blocks, rects


def getNumbersDict(template):
    digits = {}
    for name in os.listdir(template):
        img = cv2.imread(os.path.join(template, name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 150))
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化
        digits[int(name.split('.')[0])] = 255 - thresh

    return digits


if __name__ == '__main__':
    origin_card = cv2.imread(args['image'])
    card = img_resize(origin_card, args["chigh"])
    if args["imgshow"]:
        cv_show(card, 'card')
    blocks, blockRects = getCardNumbersBlocks(card)

    digits = getNumbersDict(args["template"])
    numbersBlock = []
    splitNumbers = []
    totalRects = []
    for block in blocks:
        numbers = ""
        img = block.copy()
        h, w = img.shape[:2]
        img[:, : w//4] = morphClose(img[:, : w//4], 3, 3, 3)
        img[:, w // 4:w // 2] = morphClose(img[:, w // 4: w // 2], 3, 3, 3)
        img[:, : w // 4] = morphClose(img[:, : w // 4], 3, 3, 3)
        img[:, : w // 4] = morphClose(img[:, : w // 4], 3, 3, 3)
        rects = getRect(img, 0)
        totalRects += rects
        imgs = splitByRect(img, rects)
        for img in imgs:
            number = templateMatch(img, digits)
            numbers += number
            splitNumbers.append(number)

        numbersBlock.append(numbers)

    if args["imgshow"]:
        plot_images(2, 2, blocks, numbersBlock)

    drawCard = drawNumbersBack(card, splitNumbers, totalRects, blockRects,0.6)
    cv_show(drawCard, 'card_numbers')

