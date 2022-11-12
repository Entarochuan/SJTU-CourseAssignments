#coding=utf8
#!/usr/bin/env python3
import cv2
import sys
import numpy as np
import math
import matplotlib.pyplot as plt


def Conv2d(image, w, mode='replicate'):   # Reference : https://blog.csdn.net/CV_YOU/article/details/99699307
    w = np.array(w)
    # 卷积核反转
    w = np.fliplr(np.flipud(w))

    x, y = w.shape
    image_h, image_w = image.shape

    nh = image_h + x - 1
    nw = image_w + y - 1

    # 填充大小
    add_h = int(x) // 2
    add_w = int(y) // 2

    # 填充边界
    n = np.zeros((nh, nw))
    g = np.zeros((image_h, image_w))

    # 复制原图
    n[add_h:nh - add_h, add_w:nw - add_w] = image

    if mode == 'replicate':
        # padding
        n[0:add_h, add_w:nw - add_w] = image[0, :]
        n[nh - add_h:, add_w:nw - add_w] = image[-1, :]

        for i in range(add_w):
            n[:, i] = n[:, add_w]
            n[:, nw - 1 - i] = n[:, nw - 1 - add_w]
        # 卷积运算
        for i in range(image_h):
            for j in range(image_w):
                g[i, j] = np.sum(n[i:i + x, j:j + y] * w)
        g = g.clip(0, 255)
        return g

    if mode == 'zero':
        for i in range(image_h):
            for j in range(image_w):
                g[i, j] = np.sum(n[i:i + x, j:j + y] * w)
        g = g.clip(0, 255)
        return g
    else:
        raise Exception("type error")


def sobel(operator_type, ksize=3):
    Sobel = []

    if ksize == 3:

        if operator_type == 'X':
            Sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        elif operator_type == 'Y':  # 定义求导方向
            Sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        else:
            print("type error")  # 如果未定义，则输出错误提醒

    elif ksize == 5:

        if operator_type == 'X':
            Sobel = np.array(
                [[-1, -2, 0, 2, 1], [-2, -3, 0, 3, 2], [-3, -5, 0, 5, 3], [-2, -3, 0, 3, 2], [-1, -2, 0, 2, 1]])
        elif operator_type == 'Y':  # 定义求导方向
            Sobel = np.array(
                [[1, 2, 3, 2, 1], [2, 3, 5, 3, 2], [0, 0, 0, 0, 0], [-2, -3, -5, -3, -2], [-1, -2, -3, -2, -1]])
        else:
            print("type error")  # 如果未定义，则输出错误提醒

    return Sobel


# 支持大小为 5x5 和 3x3 的Sobel算子
def detect_edges(image, ksize=3):
    """Find edge points in a grayscale image.

    Args:
    - image (2D uint8 array): A grayscale image.

    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """

    height, width = image.shape
    pad = ksize / 2
    Sobel_x = sobel('X', ksize=ksize)
    Sobel_y = sobel('Y', ksize=ksize)

    img_x = Conv2d(image, Sobel_x)
    img_y = Conv2d(image, Sobel_y)

    edge_image = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            A = img_x[h, w]
            B = img_y[h, w]
            edge_image[h, w] = math.sqrt(A ** 2 + B ** 2)

    edge_image = edge_image.clip(0, 255)
    return edge_image

    # print(img_x.shape)
    # print(img_y.shape)
    # raise NotImplementedError


def hough_circles(edge_image, edge_thresh, radius_values):
    """Threshold edge image and calculate the Hough transform accumulator array.

    Args:
    - edge_image (2D float array): An H x W heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): Hough transform accumulator array. Should have
        shape R x H x W.
    """

    pi = math.pi
    thetas = [pi / 16 * i for i in range(32)]  # 以pi/16为一个单位
    height, width = edge_image.shape
    thresh_edge_image = np.zeros((height, width))

    for h in range(0, height):
        for w in range(0, width):
            intense = edge_image[h, w]
            if intense >= edge_thresh:
                thresh_edge_image[h, w] = 255
            else:
                thresh_edge_image[h, w] = 0

    radius_num = len(radius_values)
    accum_array = np.zeros((radius_num, height, width))
    cnt = 0
    # 加权
    for r in radius_values:  # 遍历半径
        print('radius=', r)
        for h in range(0, height):
            for w in range(0, width):
                if thresh_edge_image[h, w] != 0:
                    for theta in thetas:  # 遍历角度
                        h_mid = int(h + r * math.cos(theta))
                        w_mid = int(w + r * math.sin(theta))
                        if h_mid<height and w_mid<width:
                            accum_array[cnt, h_mid, w_mid] = accum_array[cnt, h_mid, w_mid] + 1

        cnt = cnt + 1
    print('returned thresh_edge_img')
    print(accum_array[0:10])
    return thresh_edge_image, accum_array

    # raise NotImplementedError


def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """

    height, width, colors = image.shape
    cnt = 0
    circles = []
    circle_image = image
    for array in accum_array:
        r = radius_values[cnt]
        print(r)
        for h in range(height):
            for w in range(width):
                if array[h, w] > hough_thresh:
                    print('valid r = ', r)
                    circles.append((r, h, w))
                    circle_image = cv2.circle(circle_image, (w, h), r, (255, 0, 0))  # 画图
        cnt = cnt + 1

    print('Circled_image painting accomplished.')
    return circles, circle_image

    # raise NotImplementedError


def main(argv):
    img_name = argv[0]
    kernel_size, edge_thresh, hough_thresh = int(argv[1]), int(argv[2]), int(argv[3])

    # 半径检索范围
    radius_values = np.arange(10, 40)
    print(radius_values)

    # 处理阶段
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_image = detect_edges(gray_image, ksize=kernel_size)
    hist = np.bincount(edge_image.ravel().astype('int64'), minlength=256)  # 观察直方图， 选择 thresh_val = (100, 130)
    print(hist)
    # plt.figure()
    # plt.hist(edge_image.ravel(), 256, [0, 256])
    # plt.show()
    thresh_edge_image, accum_array = hough_circles(edge_image, edge_thresh, radius_values)
    circes, circle_image = find_circles(img, accum_array, radius_values, hough_thresh)

    # 结果存储与输出
    cv2.imwrite('output/p2/' + img_name + " ksize = " + str(kernel_size) + "_edge detected.png", edge_image)
    cv2.imwrite('output/p2/' + img_name + " ksize = " + str(kernel_size) + "_edge thresh img .png", thresh_edge_image)
    cv2.imwrite('output/p2/' + img_name + " ksize = " + str(kernel_size) + "_circled_image .png", circle_image)


if __name__ == '__main__':
    main(sys.argv[1:])
