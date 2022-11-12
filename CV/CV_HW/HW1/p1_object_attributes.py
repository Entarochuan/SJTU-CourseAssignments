#coding=utf8
#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import math


class Label:

    def __init__(self, num, point=None, valid=True):  # point以元组的形式输入
        if point:
            self.points = [point]
        else:
            self.points = []

        self.num = num
        self.valid = valid
        self.equal = []

    def Is_in_label(self, point):
        if point in self.points:
            return True
        return False

    def Emerge(self, points):
        self.points.extend(points)

    def Add(self, point):
        self.points.append(point)


class Component:

    def __init__(self, Intense=0, point=None):
        self.points = []
        if point:
            self.points.append(point)

        self.position = {}
        self.Area = 0
        self.Orient = 0
        self.parameters = ()
        self.Intense = Intense
        self.Orientation = 0
        self.Roundness = 0

    def Add(self, point):
        self.points.append(point)

    def Compute(self, Xy_image):   # 计算面积, 位置, Second Moments, Orientation, Roundness

        x_sum = 0
        y_sum = 0
        a, b, c = 0, 0, 0

        # 计算圆度的工具
        seen_points = []
        min_E = 100000
        max_E = 0

        for y, x in self.points:
            seen_points.append((y, x))
            for yi, xi in seen_points:
                tmp_dist = math.sqrt( (x-xi)**2 + (y-yi)**2 )
                if tmp_dist > max_E:
                    max_E = tmp_dist
                if tmp_dist < min_E and tmp_dist != 0  :
                    min_E = tmp_dist

            self.Area = self.Area + 1
            x_sum = float(x_sum + x)
            y_sum = float(y_sum + y)
            a = float(a + x ** 2)
            b = float(b + 2 * x * y)
            c = float(c + y ** 2)

        x_pos = x_sum / self.Area
        y_pos = y_sum / self.Area
        self.position['x'] = x_pos
        self.position['y'] = y_pos
        self.parameters = (a, b, c)
        self.Orientation = math.atan(b/(a-c)) / 2
        self.Roundness = max_E / min_E
        print('Computation accomplished')


def binarize(gray_image, thresh_val):
    # TODO: 255 if intensity >= thresh_val else 0
    height, width = gray_image.shape[0], gray_image.shape[1]
    binary_image = np.zeros((height, width))

    for h in range(0, height):
        for w in range(0, width):
            intense = gray_image[h, w]
            if intense >= thresh_val:
                binary_image[h, w] = 255
            else:
                binary_image[h, w] = 0

    print('returned binary')
    return binary_image


def label(binary_image):
    # TODO
    height, width = binary_image.shape[0], binary_image.shape[1]
    labeled_image = np.zeros((height, width))

    Zero_label = Label(0)
    label_list = []  # label表
    label_list.append(Zero_label)

    num = 1
    for h in range(0, height):
        for w in range(0, width):

            if binary_image[h, w] == 0:
                label_list[0].Add((h, w))

            else:  # 当label非0时

                if h == 0:  # 第一行
                    if w == 0:
                        labeled_image[h, w] = num
                        tmp_label = Label(num, (h, w))
                        label_list.append(tmp_label)
                        num = num + 1
                    else:  # 看左边点是否有标注
                        if labeled_image[h, w - 1] != 0:
                            labeled_image[h, w] = labeled_image[h, w - 1]
                            tmp_label = label_list[int(labeled_image[h, w - 1])]
                            tmp_label.Add((h, w))
                        else:
                            labeled_image[h, w] = num
                            tmp_label = Label(num, (h, w))
                            label_list.append(tmp_label)
                            num = num + 1

                else:
                    if h >= 1 and w >= 1 and labeled_image[h - 1, w - 1] != 0:  # 左上
                        labeled_image[h, w] = labeled_image[h - 1, w - 1]
                        tmp_label = label_list[int(labeled_image[h - 1, w - 1])]
                        tmp_label.Add((h, w))
                        continue

                    elif (h >= 1) and (w >= 1) and (labeled_image[h - 1, w] != 0) and (labeled_image[h, w - 1] != 0):  # equal
                        labeled_image[h, w] = labeled_image[h - 1, w]
                        label_equal1 = label_list[int(labeled_image[h - 1, w])]
                        label_equal2 = label_list[int(labeled_image[h, w - 1])]

                        if label_equal1.num != label_equal2.num:  # 等效于第二遍扫描
                            label_equal1.Emerge(label_equal2.points)  # 点合并
                            label_equal2.valid = False
                            # label_equal1.equal.append(label_equal2)   # 添加等效标识
                            # label_equal2.equal.append(label_equal1)

                            for h, w in label_equal2.points:
                                labeled_image[h, w] = label_equal1.num  # 标签替换

                    elif h >= 1 and labeled_image[h - 1, w] != 0:
                        labeled_image[h, w] = labeled_image[h - 1, w]
                        tmp_label = label_list[int(labeled_image[h - 1, w])]
                        tmp_label.Add((h, w))

                    elif h >= 1 and labeled_image[h, w - 1] != 0:
                        labeled_image[h, w] = labeled_image[h, w - 1]
                        tmp_label = label_list[int(labeled_image[h, w - 1])]
                        tmp_label.Add((h, w))

                    else:
                        labeled_image[h, w] = num
                        tmp_label = Label(num, (h, w))
                        label_list.append(tmp_label)
                        num = num + 1

    cnt = 0
    # valid_img = np.zeros((height, width))

    for label in label_list:
        if label.valid:
            cnt = cnt + 1

    print('the picture has', cnt-1, 'Connected components')

    idx = 1
    tmp_labeled_image = np.zeros((height, width))
    for label in label_list:
        if label.valid:
            if label.num == 0:  # 以0作为底色
                for (h, w) in label.points:
                    tmp_labeled_image[h, w] = 0
            else:
                for (h, w) in label.points:
                    # print(labeled_image[h, w])
                    # labeled_image[h, w] = int(idx * 255 / cnt)  # 均分实现区分度
                    tmp_labeled_image[h, w] = int(idx * 255 / cnt)
                idx = idx + 1

    return tmp_labeled_image


def get_attribute(labeled_image):
    """思路:第一次遍历找到各个部分，第二次确定信息并填表"""
    # TODO
    attribute_list = []
    height, width = labeled_image.shape[0], labeled_image.shape[1]
    Xy_image = np.zeros((height, width))

    cnt = 0  #先遍历修改为x, y形式，并顺便统计component个数,建立Component类
    exist_component = {}
    component_key_list= []

    for h in range(0, height):
        for w in range(0, width):

            y = height - h - 1
            x = w
            Xy_image[y, x] = labeled_image[h, w]

            # 添加到连通分量中
            # and not (labeled_image[h, w] in exist_component):
            if Xy_image[y, x] != 0:

                if not (Xy_image[y, x] in exist_component):  # 新分量
                    cnt = cnt + 1
                    new_component = Component(Xy_image[y, x], (y, x))
                    exist_component[int(Xy_image[y, x])] = new_component
                    component_key_list.append(Xy_image[y, x])

                else:
                    tmp_component = exist_component[Xy_image[y, x]]
                    tmp_component.Add((y, x))

    print('In labeled image, there exists', cnt, 'Connected Components.')
    print('When computing the area, assume that every point has a Binary value.')
    # print(type(exist_component))
    # 调用类内实现的计算
    comp_dict = {}
    for (Intense, component) in exist_component.items():
        # print(component.Intense)
        component.Compute(Xy_image)
        comp_dict['position'] = component.position
        comp_dict['orientation'] = component.Orientation
        comp_dict['roundness'] = component.Roundness
        # print(component.position)
        attribute_list.append(comp_dict)
        comp_dict = {}
    # print(exist_component)

    return attribute_list


def main(argv):
    img_name = argv[0]
    thresh_val = int(argv[1])
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary_image = binarize(gray_image, thresh_val=thresh_val)
    labeled_image = label(binary_image)
    attribute_list = get_attribute(labeled_image)

    cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
    cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
    cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
    print(attribute_list)


if __name__ == '__main__':
    main(sys.argv[1:])
