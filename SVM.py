import numpy as np
import random


class OptStruct:
    """
    数据结构，维护所有需要操作的值
    :parameter :
        data_mat_in - 数据矩阵
        class_labels - 数据标签
        C - 松弛变量
        tolerate - 容错率
        k_tup -- 包含核函数信息的元组，第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
    """

    def __init__(self, data_mat_in, class_labels, C, tolerate, k_tup):
        # 数据矩阵X
        self.X = data_mat_in
        # 标签数据
        self.label_mat = class_labels
        # 松弛变量
        self.C = C
        # 容错率
        self.tolerate = tolerate
        # 矩阵行数
        self.m = np.shape(data_mat_in)[0]
        # 初始化alpha、b参数
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 根据矩阵行数初始化误差缓存，第一列为是否有效的标志位，第二位为实际误差E的值
        self.eCache = np.mat(np.zeros((self.m, 2)))
        # 初始化核K
        self.K = np.mat(np.zeros((self.m, self.m)))
        # 计算所有数据的核K, K[x1, x2]低维运算得到先映射到高维再点积的高维运算结果
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], k_tup)


def kernel_trans(X, A, k_tup):
    """
    通过核函数将数据转换至更高维的空间
    :param X:  -- 数据矩阵
    :param A:  -- 单个数据的向量
    :param k_tup: -- 包含核函数信息的元组
    :return: K - 计算的核K
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    # 线性函数只进行内积
    if k_tup[0] == 'lin':
        K = X * A.T
    # 高斯核函数根据高斯核函数公式进行计算
    elif k_tup[0] == 'rbf':
        # 对于矩阵中的每一个元素计算高斯函数的值
        for j in range(m):
            # 计算核函数的分子(x - y)^2
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        # 计算高斯核K
        K = np.exp(K / (-1 * k_tup[1] ** 2))
    else:
        # 抛出错误，可以通过raise显示引发异常。一旦执行了raise语句，raise后面的语句将不能执行
        raise NameError('核函数无法识别')
    return K


def cal_Ek(oS, k):
    """
    计算误差
    :param oS: 数据结构
    :param k: 标号为k的数据
    :return: Ek - 标号为k的数据误差
    """
    fx_k = float(np.multiply(oS.alphas, oS.label_mat).T * oS.K[:, k] + oS.b)
    Ek = fx_k - float(oS.label_mat[k])
    return Ek


def select_j_rand(i, m):
    # 随机选择一个不等于i的j
    j = i
    while j == i:
        # random.uniform(x, y)方法将随机生成一个实数，它在 [x,y） 范围内
        # 不能用radiant()！！！生成[x, y]范围的整数
        j = int(random.uniform(0, m))
    return j


def select_j(oS, i, Ei):
    """
    内循环j的选取——启发方式 + 随机选择
    :param oS:  数据结构
    :param i:  标号为i的数据的索引值
    :param Ei:  标号为i的数据误差
    :return:
        j, max_k - 标号为j或max_k的数据索引值
        Ej - 标号为j的数据误差
    """
    # 初始化
    max_k = -1
    max_delta_e = 0
    Ej = 0
    # 根据Ei更新误差缓存
    oS.eCache[i] = [1, Ei]
    # 返回误差不为零的数据的索引值
    # 矩阵操作.A表示把矩阵转换为数组array
    # 存储有效误差的行索引——eCache的第一列的不为零的行下标
    valid_eCache_list = np.nonzero(oS.eCache[:, 0].A)[0]
    # 若有不为零的误差，则遍历找到最大的Ek
    if len(valid_eCache_list) > 1:
        for k in valid_eCache_list:
            # 不计算i
            if k == i:
                continue
            Ek = cal_Ek(oS, k)
            delta_e = abs(Ei - Ek)
            # 找到使|Ei - Ek|最大的Ek
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                Ej = Ek
        return max_k, Ej
    # 没有不为零的误差，则随机选择alpha_j的索引
    else:
        j = select_j_rand(i, oS.m)
        Ej = cal_Ek(oS, j)
    return j, Ej


def update_Ek(oS, k):
    """
    计算Ek,并更新误差缓存
    :param oS:  数据结构
    :param k:  标号为k的数据的索引值
    :return: 无
    """
    Ek = cal_Ek(oS, k)
    oS.eCache[k] = [1, Ek]


def clip_alpha(aj, H, L):
    # 修剪alpha
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def inner_L(i, oS):
    """
    优化的SMO算法
    :param i:  标号为i的数据的索引值
    :param oS:  数据结构
    :return:
        1 -- 有任意一对alpha值发生变化
        0 -- 没有任意一对alpha值发生变化或变化太小
    """
    # 计算误差Ei
    Ei = cal_Ek(oS, i)
    # 优化alpha,设定一定的容错率
    if ((oS.label_mat[i] * Ei < -oS.tolerate) and (oS.alphas[i] < oS.C)) \
            or ((oS.label_mat[i] * Ei > oS.tolerate) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式选择j并计算Ej
        j, Ej = select_j(oS, i, Ei)
        # 保存更新前的alpha值
        alpha_i_old = oS.alphas[i].copy()
        alpha_j_old = oS.alphas[j].copy()
        # 计算上下界L和H
        if oS.label_mat[i] != oS.label_mat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L == H")
            return 0
        # 计算eta,直接利用核函数进行低维运算
        eta = 2.0 * oS.K[i, j] - oS.K[j, j] - oS.K[i, i]
        if eta >= 0:
            # print("eta >= 0")
            return 0
        # 更新alpha_j
        oS.alphas[j] -= oS.label_mat[j] * (Ei - Ej) / eta
        # 修剪alpha_j
        oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
        # 更新Ej至缓存误差
        update_Ek(oS, j)
        if abs(oS.alphas[j] - alpha_j_old) < 0.00001:
            # print("alpha_j变化太小")
            return 0
        # 更新alpha_i
        oS.alphas[i] += oS.label_mat[j] * oS.label_mat[i] * (alpha_j_old - oS.alphas[j])
        # 更新Ei至缓存误差
        update_Ek(oS, i)
        # 更新b_1, b_2，利用核函数
        b1 = oS.b - Ei - oS.label_mat[i] * (oS.alphas[i] - alpha_i_old) * oS.K[i, i] \
            - oS.label_mat[j] * (oS.alphas[j] - alpha_j_old) * oS.K[i, j]
        b2 = oS.b - Ej - oS.label_mat[i] * (oS.alphas[i] - alpha_i_old) * oS.K[i, j] \
            - oS.label_mat[j] * (oS.alphas[j] * alpha_j_old) * oS.K[j, j]
        # 根据b_1, b_2更新b
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        # 成功更新一对alpha返回1
        return 1
    else:
        return 0


def smo_P(data_mat_in, class_labels, C, tolerate, max_iter, k_tup=('lin', 0)):
    """
    完整的线性SMO算法
    :param data_mat_in: 数据矩阵
    :param class_labels: 数据标签
    :param C: 松弛变量
    :param tolerate: 容错率
    :param max_iter: 最大迭代次数
    :param k_tup: 包含核函数信息的元组
    :return:
        oS.b -- SMO算法中计算的b
        oS.alphas -- SMO算法计算中的alphas
    """
    # 初始化
    oS = OptStruct(np.mat(data_mat_in), np.mat(class_labels).transpose(), C, tolerate, k_tup)
    iter_num = 0
    entire_set = True
    alpha_pair_changed = 0
    # 外循环条件——达到最大循环次数、内循环返回1，即有一对alpha成功被优化更新
    while (iter_num < max_iter) and ((alpha_pair_changed > 0) or entire_set):
        alpha_pair_changed = 0
        # 若还需优化，则遍历整个数据集进行优化
        if entire_set:
            for i in range(oS.m):
                alpha_pair_changed += inner_L(i, oS)
                # print(f'全样本遍历： 第{iter_num}次迭代，样本： {i}, alpha优化次数: {alpha_pair_changed}')
            iter_num += 1
        # entire_set = False
        else:
            # 存储非边界alpha的行索引信息
            non_bound_is = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in non_bound_is:
                alpha_pair_changed += inner_L(i, oS)
                # print(f'非边界遍历：第{iter_num}次迭代 样本：{i}, alpha优化次数： {alpha_pair_changed}')
            iter_num += 1
        # 实现交替遍历全体数据集和边界alpha进行更新
        # 若遍历完整个数据集，则使entire_set = False,继续更新非边界alpha
        # 若遍历完整个数据集后无优化alpha，则说明已收敛，使entire_set = False 后可提前结束循环
        if entire_set:
            entire_set = False
        # 若更新非边界alpha且无优化，则使entire_set = True，继续更新整个数据集
        elif alpha_pair_changed == 0:
            entire_set = True
        # print(f'迭代次数:{iter_num}')
    return oS.b, oS.alphas








