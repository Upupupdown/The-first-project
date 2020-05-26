import operator

import numpy as np


def load_data(filename):
    """
    数据加载函数
    :param filename: 数据文件名
    :return:
        data_mat - 加载处理后的数据集
        label_mat - 加载处理后的标签集
    """
    num_feat = len(open(filename).readline().split(';')) - 1
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split(';')
        # 循环特征数据加入line_arr
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        level = 0
        # 将分数分成3个等级
        if 0 <= float(cur_line[-1]) <= 4:
            level = 1
        elif 4 < float(cur_line[-1]) <= 7:
            level = 2
        elif 7 < float(cur_line[-1]) <= 10:
            level = 3
        label_mat.append(level)
    return np.array(data_mat), label_mat


def classify(test, data_set, label, k):
    """
    kNN算法
    :param test: 待分类的数据
    :param data_set: 已分好类的数据集
    :param label: 分类标签
    :param k: kNN算法参数，选择距离最小的k个数据
    :return: classify_result —— kNN算法分类结果
    """
    # 计算两组数据的欧氏距离
    test_copy = np.tile(test, (data_set.shape[0], 1)) - data_set
    # 二维特征相减后平方
    sq_test_copy = test_copy ** 2
    # sum() 所有元素相加，sum(0)列相加，sum(1)行相加
    row_sum = sq_test_copy.sum(axis=1)
    # 开方，得到数据点间的距离
    distance = row_sum ** 0.5
    # 返回 distances 中元素从小到大排序后的索引值
    sorted_index = distance.argsort()
    # 定义一个记录类别次数的字典
    class_count = {}
    # 遍历距离最近的前n个数据，统计类别出现次数
    for v in range(k):
        # 取出前 k 个元素的类别
        near_data_label = label[sorted_index[v]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        class_count[near_data_label] = class_count.get(near_data_label, 0) + 1
    # 根据字典的值进行降序排序
    classify_result = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别，即待分类点的类别
    return classify_result[0][0]


my_data, my_label = load_data('red_wine')
character = " "
label_arr = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
while character != "q":
    test_set = []
    for i in range(11):
        print(f'{label_arr[i]}: ', end=' ')
        character = input()
        if character != 'q':
            test_set.append(float(character))
        else:
            print("Exit!")
            break
    if character == 'q':
        break
    result = classify(np.array(test_set), my_data[1:, :], my_label, 4)
    if result == 1:
        print("Red Wine with poor quality")
    elif result == 2:
        print("Red Wine with standard quality")
    else:
        print("Red Wine with high quality")
