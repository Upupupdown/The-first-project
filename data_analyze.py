import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from SVM import*


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
            for i in range(num_feat + 1):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            if cur_line[-1] == '5':
                label_mat.append(-1)
            else:
                label_mat.append(1)
    return data_mat, label_mat


def plot_data(filename, row1, row2):
    """
    对5、6评分影响较大的特征数据两两进行分析
    :param filename: 文件名
    :param row1: 第一个特征列下标
    :param row2: 第二个特征列下标
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 可视化数据集并利用最佳参数画出决策边界
    data_mat, label_mat = load_data(filename)
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    # 记录待分析特征列下标及对应名称
    feat_name = {0: "固定酸", 1: "挥发酸", 10: "酒精浓度", 6: "二氧化硫总浓度", 7: "密度"}
    # 记录正样本
    x_cord1 = []
    y_cord1 = []
    # 记录负样本
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        # 1表示正样本， -1表示负样本， 第一列为x1， 第二列为x2
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, row1])
            y_cord1.append(data_arr[i, row2])
        else:
            x_cord2.append(data_arr[i, row1])
            y_cord2.append(data_arr[i, row2])
    # 绘制图像
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # marker='s'， 区别于负样本，用正方形绘制
    # alpha=.5 表示透明度0.5
    ax.scatter(x_cord1, y_cord1, s=20, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=20, c='green')
    plt.title('DataSet')
    plt.xlabel(feat_name[row1], FontProperties=font)
    plt.ylabel(feat_name[row2], FontProperties=font)
    plt.show()


def plot():
    plot_data('red_wine', 0, 10)
    plot_data('red_wine', 0, 1)
    plot_data('red_wine', 0, 6)
    plot_data('red_wine', 0, 7)
    plot_data('red_wine', 1, 10)
    plot_data('red_wine', 1, 6)
    plot_data('red_wine', 1, 7)
    plot_data('red_wine', 6, 10)
    plot_data('red_wine', 7, 10)
    plot_data('red_wine', 6, 7)


def test_rbf(filename, k1=25, horatio=0.1):
    """
    测试函数
    :param horatio: 测试集比例
    :param filename: 文件名
    :param k1: 使用高斯核函数的时候表示到达率
    :return: 无
    """
    data_arr, label_arr = load_two_feat(filename, 1, 10)
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


def load_two_feat(filename, row1, row2):
    """
    获得只含两个研究特征的数据集
    :param filename: 文件名
    :param row1: 第一个特征列下标
    :param row2: 第二个特征列下标
    :return: 处理好的数据集
    """
    data_arr, label_arr = load_data(filename)
    data_mat = np.array(data_arr)
    m = np.shape(data_mat)[0]
    new_data = np.zeros((m, 2))
    new_data[:, 0] = data_mat[:, row1]
    new_data[:, 1] = data_mat[:, row2]
    return new_data.tolist(), label_arr


plot()
test_rbf('red_wine')

