import numpy as np


def bin_split_data_set(data_set, feature, value):
    """
    二分切分集合
    :param data_set: 数据集
    :param feature: 待切分的特征
    :param value: 特征下的某个值
    :return: mat0, mat1 - 切分后的两个集合
    """
    # np.nonzero(data_set[:,feature] > value)[0] 返回feature值 大于 value 的行号
    mat0 = data_set[np.nonzero(data_set[:, feature] > value)[0], :]
    mat1 = data_set[np.nonzero(data_set[:, feature] <= value)[0], :]
    return mat0, mat1


def linear_solve(data_set):
    """
    线性回归模型拟合函数
    :param data_set: 数据集
    :return:
        ws - 回归系数
        X - 格式化后的特征数据集
        Y - 格式化后的标签集
    """
    m, n = np.shape(data_set)
    # 格式化X,Y中的数据
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = data_set[:, 0:n - 1]
    Y = data_set[:, -1]
    xTx = X.T * X
    # 判断是否矩阵可逆
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def reg_leaf(data_set):
    """
    生成叶节点，得到叶节点的模型；在回归树中，该模型其实就是目标变量的均值
    :param data_set: 数据集
    :return: 目标变量均值
    """
    return np.mean(data_set[:, -1])


def reg_err(data_set):
    """
    误差估计函数
    :param data_set: 数据集
    :return: 目标函数的平方误差
    """
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]


def model_leaf(data_set):
    """
    得到叶节点模型，在模型树中，该模型就是线性回归系数
    :param data_set:数据集
    :return: 回归系数ws
    """
    ws, X, Y = linear_solve(data_set)
    return ws


def model_err(data_set):
    """
    对模型树，计算误差
    :param data_set: 数据集
    :return: 误差
    """
    ws, X, Y = linear_solve(data_set)
    # 预测值
    y_hat = X * ws
    return np.sum(np.power(Y - y_hat, 2))


def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """
    找到数据的最佳二元切分方式
    :param data_set: 数据集
    :param leaf_type: 建立的叶结点函数
    :param err_type: 误差计算函数
    :param ops: 包含树构建所需的其他参数元组
    :return:
        best_index - 最佳切分行下标
        best_value - 最佳切分值
    """
    # 容许的最低误差下降值
    tol_S = ops[0]
    # 切分的最少样本数
    tol_N = ops[1]
    # 若剩余特征值数目为1，则不需要再切分，直接返回
    # 记得先将matrix转换为列表用set
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)
    m, n = np.shape(data_set)
    # 记录数据集初始误差值
    S = err_type(data_set)
    # 设置最小误差为无穷大
    best_S = float('inf')
    best_index = 0
    best_value = 0
    # 遍历数据集中的每一个特征
    for feat_index in range(n - 1):
        # 遍历特征下的每一个值作为切分值
        for split_val in set(data_set[:, feat_index].T.tolist()[0]):
            # 根据特征及特征值进行划分数据集
            mat0, mat1 = bin_split_data_set(data_set, feat_index, split_val)
            # 若划分后的任一集合样本数少于给定切分最少样本数，则继续下一个切分值
            if np.shape(mat0)[0] < tol_N or np.shape(mat1)[0] < tol_N:
                continue
            # 否则计算出新的误差值
            new_S = err_type(mat0) + err_type(mat1)
            # 保存最小误差值下的信息
            if new_S < best_S:
                best_index = feat_index
                best_value = split_val
                best_S = new_S
    # 若经历循环后，误差值的变化小于设定的最低误差下降值，则返回none，并直接创建叶节点
    if S - best_S < tol_S:
        return None, leaf_type(data_set)
    mat0, mat1 = bin_split_data_set(data_set, best_index, best_value)
    # 检验切分后的子集大小，若某个子集大小小于给定的最少切分样本数，则不切分，返回none，直接创建叶节点
    if np.shape(mat0)[0] < tol_N or np.shape(mat1)[0] < tol_N:
        return None, leaf_type(data_set)
    # 若成功划分，则返回最佳划分下标和最佳划分值
    return best_index, best_value


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """
    创建回归（分类）树
    :param data_set: 数据集
    :param leaf_type: 建立叶节点函数
    :param err_type: 误差计算函数
    :param ops: 树构建所需要的其他参数元组
    :return:
        ret_tree - 构建好的回归树
    """
    # 选择最佳划分特征及特征值
    feat, val = choose_best_split(data_set, leaf_type, err_type, ops)
    # 若不能划分，则直接返回创建的叶节点
    if feat is None:
        return val
    # 构建回归树
    ret_tree = {'spInd': feat, 'spVal': val}
    # 得到划分后的左右子树
    l_set, r_set = bin_split_data_set(data_set, feat, val)
    # 递归进行构建左右子树
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


def is_tree(obj):
    """
    判断输入的变量是否为一棵树
    :param obj: 输入变量
    :return: 判断的布尔类型变量结果
    """
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    """
    从上往下遍历树知道叶节点为止，找到两个叶节点则计算它们的平均值
    :param tree: 回归树
    :return: 两个叶节点的均值
    """
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_data):
    """
    回归树剪枝函数
    :param tree: 待剪枝的树
    :param test_data: 剪枝所需要的测试集
    :return: 剪枝后的回归树
    """
    # 判断（划分后的）测试集是否为空，若为空，则返回均值
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)
    # 否则，若左右子树不全为叶子节点，则切分测试集
    if is_tree(tree['left']) or is_tree(tree['right']):
        # 利用训练集得到的回归树划分当前测试集
        l_set, r_set = bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
    # 左分支为子树，则递归剪枝
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], l_set)
    # 右分支为子树，则递归剪枝
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], r_set)
    # 若左右分支均为叶节点，则比较误差
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        # 划分当前测试集
        l_set, r_set = bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
        # 计算未进行合并叶节点（塌陷处理）的误差值
        error_no_merge = np.sum(np.power(l_set[:, -1] - tree['left'], 2)) \
                         + np.sum(np.power(r_set[:, -1] - tree['right'], 2))
        # 计算合并的叶节点的均值
        tree_mean = (tree['left'] + tree['right']) / 2.0
        # 计算合并后的误差
        error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        # 若合并后误差减小则返回剪枝后的树
        if error_merge < error_no_merge:
            return tree_mean
        # 否则不进行剪枝直接返回
        else:
            return tree
    else:
        return tree


def reg_tree_eval(model, in_dat):
    # 返回回归树的预测值
    return float(model)


def model_tree_eval(model, in_dat):
    # 格式化测试数据集计算模型预测结果
    n = np.shape(in_dat)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = in_dat
    return float(X * model)


def tree_forecast(tree, in_data, model_eval=reg_tree_eval):
    # 预测，自顶向下遍历整棵树，直到叶节点调用函数预测值
    if not is_tree(tree):
        return model_eval(tree, in_data)
    if in_data.tolist()[0][tree['spInd']] > tree['spVal']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'], in_data, model_eval)
        else:
            return model_eval(tree['left'], in_data)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], in_data, model_eval)
        else:
            return model_eval(tree['right'], in_data)


def create_forecast(tree, test_data, model_eval=reg_tree_eval):
    # 对测试集中的所有样例进行预测返回预测值列表
    m = len(test_data)
    y_hat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_forecast(tree, np.mat(test_data[i]), model_eval)
    return y_hat


