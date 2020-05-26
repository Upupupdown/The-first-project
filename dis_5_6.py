import operator
from SVM import*
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
        # 得到分数为5和6的数据集
        if cur_line[-1] == '5' or cur_line[-1] == '6':
            # 循环特征数据加入line_arr
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            if cur_line[-1] == '5':
                label_mat.append(-1)
            else:
                label_mat.append(1)
    return data_mat, label_mat


# 利用kNN模型对数据集进行分类
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
    # print(classify_result)
    # 返回次数最多的类别，即待分类点的类别
    return classify_result[0][0]


def test_for_kNN(filename, horatio=0.1, k=4):
    """
    利用kNN对5、6分数类别进行分类
    :param filename: 文件名
    :param horatio: 测试集比例
    :param k: kNN参数
    :return: 无
    """
    data, label = load_data(filename)
    data = np.array(data)
    m = np.shape(data)[0]
    test_num = int(m * horatio)
    error_count = 0.0
    for i in range(test_num):
        classify_result = classify(data[i, :], data[test_num:m, :], label[test_num:m], k)
        if classify_result != label[i]:
            error_count += 1.0
    print("kNN模型的预测准确率： %.1f%%" % (100 * (1 - error_count / test_num)))


# 利用支持向量机模型对数据集进行分类
def test_rbf(filename, k1=20, horatio=0.1):
    """
    测试函数
    :param horatio: 测试集比例
    :param filename: 文件名
    :param k1: 使用高斯核函数的时候表示到达率
    :return: 无
    """
    data_arr, label_arr = load_data(filename)
    m = len(data_arr)
    test_num = int(m * horatio)
    test_arr = data_arr[0:test_num]
    test_label = label_arr[0:test_num]
    train_arr = data_arr[test_num:]
    train_label = label_arr[test_num:]
    b, alphas = smo_P(train_arr, train_label, 200, 0.0001, 100, ('rbf', k1))
    train_mat = np.mat(train_arr)
    train_label_mat = np.mat(train_label).transpose()
    # 获得支持向量
    sv_ind = np.nonzero(alphas.A > 0)[0]
    svs = train_mat[sv_ind]
    label_sv = train_label_mat[sv_ind]
    print(f'支持向量个数: {np.shape(svs)[0]}')
    m, n = np.shape(train_mat)
    error_count = 0
    for i in range(m):
        # 计算各点的核
        kernel_eval = kernel_trans(svs, train_mat[i, :], ('rbf', k1))
        # 根据支持向量的点，计算超平面，返回预测结果
        predict = kernel_eval.T * np.multiply(label_sv, alphas[sv_ind]) + b
        # 返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
        if np.sign(predict) != np.sign(train_label[i]):
            error_count += 1
    print(f'训练集准确率：{(1 - float(error_count) / m) * 100}')
    # 加载测试集
    error_count = 0
    test_mat = np.mat(test_arr)
    m, n = np.shape(test_mat)
    for i in range(m):
        kernel_eval = kernel_trans(svs, test_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * np.multiply(label_sv, alphas[sv_ind]) + b
        if np.sign(predict) != np.sign(test_label[i]):
            error_count += 1
    print(f'测试集准确率:{(1 - float(error_count) / m) * 100}')


# 利用AdaBoost模型对数据集进行分类
def stump_classify(data_matrix, col, thresh_val, thresh_flag):
    """
    单层决策树分类函数
    :param data_matrix: 数据矩阵
    :param col: 第cal列，也就是第几个特征
    :param thresh_val: 阈值
    :param thresh_flag: 标志
    :return:
        ret_array - 分类结果
    """
    # 初始化预测分类结果
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    if thresh_flag == 'lt':
        # col列的特征数据小于（'lt'）分界值（阈值 thresh_val）时，将其类别设置为负类，值为-1.0（基于某阈值的预测）
        ret_array[data_matrix[:, col] <= thresh_val] = -1.0
    else:
        # col列的特征数据大于（'gt'）分界值（阈值 thresh_val）时，将其类别设置为负类，值为-1.0（基于某阈值的预测）
        ret_array[data_matrix[:, col] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, D):
    """
    找到数据集上最佳的单层决策树
    :param data_arr: 数据矩阵
    :param class_labels: 数据标签
    :param D: 样本权重
    :return:
        best_stump - 最佳单层决策树信息
        min_error - 最小误差
        best_result - 最佳分类结果
    """
    data_matrix = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_matrix)
    num_steps = 10.0
    best_stump = {}
    best_result = np.mat(np.zeros((m, 1)))
    # 初始化最小错误率为无穷大
    min_error = float('inf')
    # 遍历不同特征（遍历列）
    for i in range(n):
        # 找出特征数据极值（一列中的最大和最小值），设置步长 step_size（即增加阈值的步长）
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps
        # 设置不同的阈值，计算以该阈值为分界线的分类结果 ———— 不同阈值不同分类情况（'lt', 'gt'）找到错误率最小的分类方式
        # 阈值的设置从最小值-步长到最大值，以步长为间隔逐渐增加阈值
        # 分类结果设置按小于（'lt'）阈值为负类和大于（'gt'）阈值为负类分别进行设置，计算最后分类结果
        for j in range(-1, int(num_steps) + 1):
            for situation in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_val = stump_classify(data_matrix, i, thresh_val, situation)
                err_arr = np.mat(np.ones((m, 1)))
                # 将分类正确的设置为0
                err_arr[predicted_val == label_mat] = 0
                # 计算错误率
                weighted_error = D.T * err_arr
                # print('\n split:dim %d, thresh %.2f, thresh situation: %s \
                # the weighted error is %.3f' % (i, thresh_val, situation, weighted_error))
                # 记录最小错误率时的信息，生成最佳单层决策树
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_result = predicted_val.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['situation'] = situation
    return best_stump, min_error, best_result


def ada_boost_train_DS(data_arr, class_labels, num_iter=40):
    """
    基于单层决策树的AdaBoost训练
    :param data_arr: 数据集
    :param class_labels: 数据标签
    :param num_iter: 迭代次数
    :return:
        weak_class_arr - 多次训练后得到的弱分类器
    """
    # 存放分类器提升过程中的弱分类器
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    # 初始化权重
    D = np.mat(np.ones((m, 1)) / m)
    agg_class_result = np.mat(np.zeros((m, 1)))
    for i in range(num_iter):
        # 构建单层决策树
        # 弱分类器的错误率 error -> 分类器的权重 alpha -> 数据类别结果权重 -> 弱分类器错误率
        # 弱分类器的错误率 error -> 分类器的权重 alpha -> 累计结果估计值 agg_class_result -> 为0时结束训练
        best_stump, error, class_result = build_stump(data_arr, class_labels, D)
        # print(D.T)
        # 计算alpha,为每个分类器分配的一个权重值alpha，基于每个弱分类器的错误率进行计算
        # max(error, 1e-16)是为避免当弱分类器的错误率为零时进行除零运算
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        # 记录求得的alpha，和该弱分类器的分类结果
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        # print(f'class_result: {class_result}')
        # 计算改变样本权重的e的指数，分类正确为-alpha,错误则为alpha，利用label与result相乘判断正负
        e_exponent = np.multiply(-1 * alpha * np.mat(class_labels).T, class_result)
        D = np.multiply(D, np.exp(e_exponent))
        D = D / D.sum()
        # 记录每个数据点的类别估计积累值
        agg_class_result += alpha * class_result
        # print(f'agg_class_result: {agg_class_result.T}')
        # 计算累加错误率
        agg_errors = np.multiply(np.sign(agg_class_result) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        # print(f'total error: {error_rate}')
        if error_rate == 0.0:
            break
    return weak_class_arr


def ada_classify(test_data, classifier_arr):
    """
    测试分类函数
    :param test_data: 测试数集
    :param classifier_arr: AdaBoost训练得到的弱分类器集合
    :return:
        sign(agg_class_result) - 分类结果
    """
    test_matrix = np.mat(test_data)
    m = np.shape(test_matrix)[0]
    # 累计分类估计值
    agg_class_result = np.mat(np.zeros((m, 1)))
    # 遍历得到的每一个弱分类器，
    for i in range(len(classifier_arr)):
        # 根据该分类器进行分类
        class_result = stump_classify(test_matrix, classifier_arr[i]['dim'],
                                      classifier_arr[i]['thresh'], classifier_arr[i]['situation'])
        # 利用分类器权重累加分类估计值
        agg_class_result += classifier_arr[i]['alpha'] * class_result
        # print(agg_class_result)
    # 利用sign函数得到分类结果,其实是根据概率进行分类
    # 根据累加的估计分类值，值属于正样本的概率大（这里为值大于0），则判为正类，
    # 属于负样本的概率大（小于0），则判为负类。实质上这里的分类阈值为0.5
    return np.sign(agg_class_result)


def test_for_Ada(filename, horatio=0.1, num_item=30):
    # 利用AdaBoost算法对数据集进行训练预测
    data_arr, label_arr = load_data(filename)
    m = len(data_arr)
    # 划分数据集为训练集和测试集
    test_num = int(m * horatio)
    test_arr = data_arr[0:test_num]
    test_label = label_arr[0:test_num]
    train_arr = data_arr[test_num:]
    train_label = label_arr[test_num:]
    # 基于单层决策树训练训练集
    classifier_arr = ada_boost_train_DS(train_arr, train_label, num_item)
    # 对测试集进行预测并统计其错误率
    prediction = ada_classify(test_arr, classifier_arr)
    m = np.shape(test_arr)[0]
    error_arr = np.mat(np.ones((m, 1)))
    print("AdaBoost模型预测准确率： %.1f%%" % (100 * (1 - error_arr[prediction != np.mat(test_label).T].sum() / m)))


test_for_Ada('red_wine', horatio=0.1, num_item=35)
test_for_kNN('red_wine', k=4)
test_rbf('red_wine')
