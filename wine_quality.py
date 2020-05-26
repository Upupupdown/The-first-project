import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties


def pre_data():
    """
    获取数据
    :return: 处理好的数据集
    """
    # 数据输出精度
    pd.set_option('precision', 3)
    # 读取数据并显示出前4条数据
    dfr = pd.read_csv(r'C:\Users\惠普\Downloads\winequality-red(1).csv', sep=';')
    dfw = pd.read_csv(r'C:\Users\惠普\Downloads\winequality-white(1).csv', sep=';')
    print(dfr.head())
    # 添加属性总酸至表格首列——固定酸和挥发酸的总和
    dfr['total acidity'] = dfr['fixed acidity'] + dfr['volatile acidity']
    dfw['total acidity'] = dfw['fixed acidity'] + dfw['volatile acidity']
    r = dfr.columns.tolist()
    r.insert(0, r.pop())
    dfr = dfr.reindex(columns=r)
    r = dfw.columns.tolist()
    r.insert(0, r.pop())
    dfw = dfw.reindex(columns=r)
    # 描述统计
    print(dfr.describe())
    print(dfw.describe())
    return dfr, dfw


def pca_mat(x):
    """
    获取参与运算的多维数组并标准化数据
    :param x: CSV格式数据集
    :return: 处理好的array型数据
    """
    # 获取去除评分项及总酸特征数据的数据表
    temp = x.drop(['quality', 'total acidity'], axis=1)
    # dataframe格式转为多维数组
    XMat = np.array(temp)
    # axis=0表示按照列来求均值
    aver = np.mean(XMat, axis=0)
    # 求每列标准差
    standard = np.std(XMat, axis=0)
    # 中心标准化
    data_adjust = (XMat - aver) / standard
    return data_adjust


def pca_eig(data_adjust):
    """
    获取满足要求的前k个特征值和特征向量
    :param data_adjust: 数据集
    :return: 前k的特征值特征向量以及获得的特征值、特征向量
    """
    # 计算协方差矩阵
    covmat = np.cov(data_adjust, rowvar=False)
    # 求解协方差矩阵的特征值和特征向量
    eigVals, eigVects = np.linalg.eig(covmat)
    # 按照eigVals进行从大到小排序（给出序号，不修改原特征值列表）
    eigValInd = np.argsort(-eigVals)
    # 确定前k的主成分，使选取的主成分贡献90%以上的方差
    val_sum = 0
    val_total = eigVals.sum()
    for k in eigValInd:
        val_sum += eigVals[k]
        if val_sum / val_total < 0.90:
            continue
        else:
            break
    """分割线"""
    x = int(np.argwhere(eigValInd == k) + 1)  # 定位k所在位置，结果加1
    eigValInd = eigValInd[:x:1]  # 截取前k个特征值的序号
    """取前k特征值"""
    list = []
    for i in eigValInd:
        list.append(eigVals[i])
    redEigVals = np.array(list)
    """对应前k的特征向量"""
    redEigVects = []
    for i in eigValInd:
        redEigVects.append(eigVects[i])
    redEigVects = np.array(redEigVects).T
    return redEigVals, redEigVects, eigVals, eigVects


def pca_coe(data_adjust):
    """
    返回主成分系数
    :param data_adjust: 标准数据集
    :return: 主成分系数
    """
    return pca_eig(data_adjust)[1] / (pca_eig(data_adjust)[0] ** 0.5)


def pca(data_adjust):
    """
    返回每个样本的主成分得分
    :param data_adjust: 标准数据集
    :return: 降维后的数据集
    """
    lowDDataMat = np.mat(data_adjust) * pca_eig(data_adjust)[1]
    return lowDDataMat


def test1():
    """红、白葡萄酒初始分析数据"""
    dfr, dfw = pre_data()
    # 数据标准化
    pr = pca_mat(dfr)
    pw = pca_mat(dfw)
    print("未进行降维处理前，得到的协方差矩阵的特征值情况")
    print(pca_eig(pr)[2])
    print(pca_eig(pw)[2])
    print("90%的标准下，降维后得到的特征值")
    print(pca_eig(pr)[0])
    print(pca_eig(pw)[0])
    # 主成分系数，解释为第K个主成分表示为11个输入变量的线性组合
    print(pd.DataFrame(pca_coe(pr)))
    print(pd.DataFrame(pca_coe(pw)))
    # 主成分得分， 解释为每个样本点在主成分上投影的坐标,转换为的低维数据集
    print(pd.DataFrame(pca(pr)))
    print(pd.DataFrame(pca(pw)))
    pd.DataFrame(pca(pr)).to_csv(path_or_buf='C:\\Users\\惠普\\Documents\\low.txt', float_format='%.3f', header=False, 
                                 index=False)


test1()
