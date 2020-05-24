import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties


def pre_data():
    """
    获取csv文件（文本文件）中的数据信息
    :return: 经过处理得到的数据集
    """
    # 数据输出精度
    pd.set_option('precision', 3)
    # 读取数据并显示出前4条数据
    dfr = pd.read_csv(r'C:\Users\惠普\Downloads\winequality-red(1).csv', sep=';')
    dfw = pd.read_csv(r'C:\Users\惠普\Downloads\winequality-white(1).csv', sep=';')
    # print(dfr.head())
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
    # print(dfr.describe())
    # print(dfw.describe())
    return dfr, dfw


def plot_box(dfr, dfw):
    """
    绘制体现各特征数据分布的箱线图
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    colnm_r = dfr.columns.tolist()
    colnm_w = dfw.columns.tolist()
    plt.figure(figsize=(10, 6))
    plt.suptitle('box_compare', fontsize=14, y=1.05)  # 总标题
    """画第一行的图"""
    for i in range(7):
        y1 = dfr[colnm_r[i]]
        y2 = dfw[colnm_w[i]]
        data = pd.DataFrame({"red": y1, "white": y2})
        plt.subplot(2, 7, i + 1)
        data.boxplot(widths=0.5, flierprops={'marker': 'o', 'markersize': 2})
        plt.ylabel(colnm_r[i], fontsize=12)
    plt.tight_layout()
    """画第二行的图"""
    for i in range(6):
        y1 = dfr[colnm_r[i + 7]]
        y2 = dfw[colnm_w[i + 7]]
        data = pd.DataFrame({"red": y1, "white": y2})
        plt.subplot(2, 6, i + 7)
        data.boxplot(widths=0.5, flierprops={'marker': 'o', 'markersize': 2})
        plt.ylabel(colnm_r[i + 7], fontsize=12)
    plt.tight_layout()
    plt.show()


def fixed_total_rate(dfr, dfw):
    """
    绘制固定酸占总酸的比重的图
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    color = sns.color_palette()
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('固定酸占总酸比分布情况', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    temp = dfr[{'total acidity', 'fixed acidity'}]
    # 计算占比
    temp['percent'] = temp.apply(lambda x: x['fixed acidity'] / x['total acidity'], axis=1)
    temp['percent'].hist(bins=100, color=color[0])
    plt.xlabel('红葡萄酒固定酸占比', fontsize=12, FontProperties=font)
    plt.ylabel('频数', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    temp = dfw[{'total acidity', 'fixed acidity'}]
    # 计算占比
    temp['percent'] = temp.apply(lambda x: x['fixed acidity'] / x['total acidity'], axis=1)
    temp['percent'].hist(bins=100, color=color[0])
    plt.xlabel('白葡萄酒固定酸占比', fontsize=12, FontProperties=font)
    plt.ylabel('频数', fontsize=12, FontProperties=font)
    plt.show()


def fixed_influence_quality(dfr, dfw):
    """
    可视化分析固定酸占比对评分的影响
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('固定酸占总酸比对评分的影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    temp = dfr[{'total acidity', 'fixed acidity', 'quality'}]
    # 计算占比
    temp['precent'] = temp.apply(lambda x: x['fixed acidity'] / x['total acidity'], axis=1)
    sns.boxplot(x=temp['quality'], y=temp['precent'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('固定酸占比', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    temp = dfw[{'total acidity', 'fixed acidity', 'quality'}]
    # 计算占比
    temp['precent'] = temp.apply(lambda x: x['fixed acidity'] / x['total acidity'], axis=1)
    sns.boxplot(x=temp['quality'], y=temp['precent'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('固定酸占比', fontsize=12, FontProperties=font)
    plt.show()


def citric_influence_quality(dfr, dfw):
    """
    可视化分析柠檬酸对总酸的占比对评分的影响
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('柠檬酸占固定酸比对评分的影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    temp = dfr[{'citric acid', 'fixed acidity', 'quality'}]
    # 计算占比
    temp['precent'] = temp.apply(lambda x: x['citric acid'] / x['fixed acidity'], axis=1)
    sns.boxplot(x=temp['quality'], y=temp['precent'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('柠檬酸占比', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    temp = dfw[{'citric acid', 'fixed acidity', 'quality'}]
    # 计算占比
    temp['precent'] = temp.apply(lambda x: x['citric acid'] / x['fixed acidity'], axis=1)
    sns.boxplot(x=temp['quality'], y=temp['precent'])
    plt.xlabel('白葡萄酒评分', fontsize=10, FontProperties=font)
    plt.ylabel('柠檬酸占比', fontsize=10, FontProperties=font)
    plt.show()


def volatile_influence_quality(dfr, dfw):
    """
    可视化挥发酸占比对评分的影响
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('挥发酸占总酸比对评分的影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    temp = dfr[{'total acidity', 'volatile acidity', 'quality'}]
    # 计算占比
    temp['precent'] = temp.apply(lambda x: x['volatile acidity'] / x['total acidity'], axis=1)
    sns.boxplot(x=temp['quality'], y=temp['precent'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('挥发酸占比', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    temp = dfw[{'total acidity', 'volatile acidity', 'quality'}]
    # 计算占比
    temp['precent'] = temp.apply(lambda x: x['volatile acidity'] / x['total acidity'], axis=1)
    sns.boxplot(x=temp['quality'], y=temp['precent'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('挥发酸占比', fontsize=12, FontProperties=font)
    plt.show()


def acidity_pH_influence_quality(dfr, dfw):
    """
    可视化总酸以及PH对评分的影响
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('总酸含量对评分的影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    temp = dfr[{'total acidity', 'quality'}]
    sns.boxplot(x=temp['quality'], y=temp['total acidity'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('总酸含量', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    temp = dfw[{'total acidity', 'quality'}]
    sns.boxplot(x=temp['quality'], y=temp['total acidity'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('总酸含量', fontsize=12, FontProperties=font)
    plt.show()

    # pH对评分影响
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('pH值对评分的影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    temp = dfr[{'pH', 'quality'}]
    sns.boxplot(x=temp['quality'], y=temp['pH'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('pH值', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    temp = dfw[{'pH', 'quality'}]
    sns.boxplot(x=temp['quality'], y=temp['pH'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('pH值', fontsize=12, FontProperties=font)
    plt.show()


def residual_influence_quality(dfr, dfw):
    """
    可视化残留糖对评分的影响
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('残留糖含量对评分的影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    sns.boxplot(x=dfr['quality'], y=dfr['residual sugar'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('残留糖含量', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    sns.boxplot(x=dfw['quality'], y=dfw['residual sugar'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('残留糖含量', fontsize=12, FontProperties=font)
    plt.show()


def alcohol_influence_quality(dfr, dfw):
    """
    可视化酒精浓度对评分的影响
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.figure(figsize=(10, 4))
    plt.suptitle('酒精浓度对评分的影响', y=0.98, fontsize=8, FontProperties=font)  # 总标题
    """红"""
    plt.subplot(1, 2, 1)
    sns.boxplot(x=dfr['quality'], y=dfr['alcohol'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('酒精浓度', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    sns.boxplot(x=dfw['quality'], y=dfw['alcohol'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('酒精浓度', fontsize=12, FontProperties=font)
    plt.show()


def salt_influence_quality(dfr, dfw):
    """
    可视化氯化物浓度以及硫酸盐浓度对评分的影响
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('氯化物浓度对评分的影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    sns.boxplot(x=dfr['quality'], y=dfr['chlorides'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('氯化物浓度', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    sns.boxplot(x=dfw['quality'], y=dfw['chlorides'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('氯化物浓度', fontsize=12, FontProperties=font)
    plt.show()

    # 硫酸盐浓度对评分影响
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('硫酸盐浓度对评分的影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    sns.boxplot(x=dfr['quality'], y=dfr['sulphates'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('硫酸盐浓度', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    sns.boxplot(x=dfw['quality'], y=dfw['sulphates'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('硫酸盐浓度', fontsize=12, FontProperties=font)
    plt.show()


def sulfur_dioxide_influence_quality(dfr, dfw):
    """
    可视化游离二氧化硫以及总二氧化硫对评分的影响
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('游离二氧化硫占总二氧化硫比重对评分的影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    temp = dfr[{'free sulfur dioxide', 'total sulfur dioxide', 'quality'}]
    temp['precent'] = temp.apply(lambda x: x['free sulfur dioxide'] / x['total sulfur dioxide'], axis=1)
    sns.boxplot(x=temp['quality'], y=temp['precent'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('游离二氧化硫占比', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    temp = dfw[{'free sulfur dioxide', 'total sulfur dioxide', 'quality'}]
    temp['precent'] = temp.apply(lambda x: x['free sulfur dioxide'] / x['total sulfur dioxide'], axis=1)
    sns.boxplot(x=temp['quality'], y=temp['precent'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('游离二氧化硫占比', fontsize=12, FontProperties=font)
    plt.show()

    # 二氧化硫总量对评分影响
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('二氧化硫总量对评分的影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    sns.boxplot(x=dfr['quality'], y=dfr['total sulfur dioxide'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('二氧化硫总量', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    sns.boxplot(x=dfw['quality'], y=dfw['total sulfur dioxide'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('二氧化硫总量', fontsize=12, FontProperties=font)
    plt.show()


def density_influence_quality(dfr, dfw):
    """
    可视化密度对评分的影响
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.figure(figsize=(10, 4))
    # 总标题
    plt.suptitle('密度对评分影响', y=0.98, fontsize=8, FontProperties=font)
    """红"""
    plt.subplot(1, 2, 1)
    sns.boxplot(x=dfr['quality'], y=dfr['density'])
    plt.xlabel('红葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('密度', fontsize=12, FontProperties=font)
    """白"""
    plt.subplot(1, 2, 2)
    sns.boxplot(x=dfw['quality'], y=dfw['density'])
    plt.xlabel('白葡萄酒评分', fontsize=12, FontProperties=font)
    plt.ylabel('密度', fontsize=12, FontProperties=font)
    plt.show()


def factors_influence_quality(dfr, dfw):
    """
    可视化评分与各个特征之间的关系
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    # 红葡各变量与评分关系
    color = sns.color_palette()
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    colnm = dfr.columns.tolist()[:12]
    plt.figure(figsize=(10, 8))

    for i in range(12):
        plt.subplot(4, 3, i + 1)
        sns.boxplot(x='quality', y=colnm[i], data=dfr, color=color[1], width=0.6)
        plt.ylabel(colnm[i], fontsize=12)
    plt.suptitle('红葡萄酒各变量与评分关系--箱线图', y=1.01, fontsize=8, FontProperties=font)
    plt.tight_layout()
    plt.show()
    # 白葡各变量与评分关系
    colnm = dfw.columns.tolist()[:12]
    plt.figure(figsize=(10, 8))

    for i in range(12):
        plt.subplot(4, 3, i + 1)
        sns.boxplot(x='quality', y=colnm[i], data=dfw, color=color[1], width=0.6)
        plt.ylabel(colnm[i], fontsize=12)
    plt.suptitle('白葡萄酒各变量与评分关系--箱线图', y=1.00, fontsize=8, FontProperties=font)
    plt.tight_layout()
    plt.show()


def heat_map(dfr, dfw):
    """
    绘制热力相关图，展现各数据间的关系
    :param dfr: 红葡萄酒数据集
    :param dfw: 白葡萄酒数据集
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 红葡热力相关图
    plt.figure(figsize=(10, 8))
    colnm = dfr.columns.tolist()
    mcorr = dfr[colnm].corr()
    mask = np.zeros_like(mcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
    plt.title('红葡萄酒各变量间热力相关图', FontProperties=font)
    plt.show()
    # 白葡热力相关图
    plt.figure(figsize=(10, 8))
    colnm = dfw.columns.tolist()
    mcorr = dfw[colnm].corr()
    mask = np.zeros_like(mcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
    plt.title('白葡萄酒各变量间热力相关图', FontProperties=font)
    plt.show()


def plot():
    """
    可视化分析数据
    :return: 无
    """
    r, w = pre_data()
    plot_box(r, w)
    fixed_total_rate(r, w)
    fixed_influence_quality(r, w)
    citric_influence_quality(r, w)
    volatile_influence_quality(r, w)
    acidity_pH_influence_quality(r, w)
    residual_influence_quality(r, w)
    alcohol_influence_quality(r, w)
    salt_influence_quality(r, w)
    sulfur_dioxide_influence_quality(r, w)
    density_influence_quality(r, w)
    factors_influence_quality(r, w)
    heat_map(r, w)


plot()
