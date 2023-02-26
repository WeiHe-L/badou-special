# -*- coding:utf-8 -*-
"""
@author:Timothy

彩色图像的灰度化、二值化
"""

# Opencv基础

# 导入Opencv库
import cv2

# 以矩阵的方式读取一张图片
image = cv2.imread("lenna.png")

# 读取矩阵/图像 的长、宽(找循环的边界)
[w, h] = image.shape[:2]
# print(w, h)


# 导入numpy库
import numpy as np

# 创建一张和当前图片大小一样的单通道图片
image_gray = np.zeros([h, w], image.dtype)

# 灰度化
#  灰度化的循环实现（便于理解原理）
for i in range(w):
    for j in range(h):
        # 将当前元素点的三色按照生物加权的方式灰度处理
        image_gray[i, j] = (image[i, j][0] * 0.11 + image[i, j][1] * 0.59 + image[i, j][2] * 0.3)

# 输出 循环灰度化 后的图片矩阵
# print(image_gray)

cv2.imshow("image show gray(loop way)", image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows() # 循环实现灰度化成功！

# 展示到plt上

# 导入matplotlib
import matplotlib.pyplot as plt

# 设置到展示板上
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

#  灰度化的函数实现（实际操作中这样做）

# 导入库
from skimage.color import rgb2gray

# 函数灰度化实现
image_gray = rgb2gray(image)  # 注意：通过函数灰度化后的矩阵，矩阵元素值范围为0 -- 1.0

cv2.imshow("image show gray(functhion way)", image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 函数灰度化的图像显示在展示面板上
plt.subplot(222)
plt.imshow(image_gray, cmap='gray')


# 二值化
# 注意：二值化是在灰度化基础上实现的，要实现二值化，首先要实现灰度化
#  二值化的循环实现（便于理解原理）

# 获取灰度图(原图)行、列数
rows, cols = image_gray.shape

# 循环遍历实现图像矩阵的二值化
for i in range(rows):
    for j in range(cols):
        if image_gray[i, j] <= 0.5:
            image_gray[i, j] = 0
        else:
            image_gray[i, j] = 1

cv2.imshow("image_binary(loop way)", image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#  二值化的函数实现（实际操作中这样做）
image_binary = np.where(image_gray >= 0.5, 1, 0)
cv2.imshow("image_binary(function way)", image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 函数二值化的图像显示在展示面板上
plt.subplot(223)
plt.imshow(image_binary, cmap='binary')
plt.show()



