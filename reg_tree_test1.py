from reg_tree import*


def load_data_set(filename):
    """
    加载数据集，获取类别为分数的数据集
    :param filename: 文件名
    :return: data_mat - 数据集
    """
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split(';')
        # 将每行映射为浮点数
        flt_line = list(map(float, cur_line))
        data_mat.append(flt_line)
    return data_mat


def load_data(filename):
    """
    获取修改分数为等级后的数据集
    :param filename:
    :return:
    """
    fr = open(filename)
    data_arr = []
    # 下载数据
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split(';')
        # 循环特征数据加入line_arr
        if 0 <= float(cur_line[-1]) <= 4.0:
            cur_line[-1] = '1'
        elif 4 < float(cur_line[-1]) <= 7.0:
            cur_line[-1] = '2'
        elif 7.0 < float(cur_line[-1]) <= 10.0:
            cur_line[-1] = '3'
        for i in range(len(cur_line)):
            line_arr.append(float(cur_line[i]))
        data_arr.append(line_arr)
    return data_arr


def test1(filename):
    """
    分数预测
    :param filename: 文件名
    :return: 无
    """
    my_mat = load_data_set(filename)
    n = int(len(my_mat) * 0.9)
    m = len(my_mat[0])
    train_mat = np.mat(my_mat[0:n])
    test_mat = np.mat(my_mat[n:])
    my_tree = create_tree(train_mat, ops=(1, 18))
    print(my_tree)
    my_tree = prune(my_tree, test_mat)
    y_hat = create_forecast(my_tree, test_mat[:, 0:m-1])
    error = y_hat - test_mat[:, m-1]
    print("回归树分数预测准确率： %.1f%%." % (100 * (len(np.nonzero(abs(error) <= 0.6)[0]) / len(test_mat))))


def test2(filename):
    """
    品质等级预测
    :param filename: 文件名
    :return: 无
    """
    my_mat = load_data(filename)
    n = int(len(my_mat) * 0.9)
    m = len(my_mat[0])
    train_mat = np.mat(my_mat[0:n])
    test_mat = np.mat(my_mat[n:])
    my_tree = create_tree(train_mat, ops=(1, 18))
    print(my_tree)
    my_tree = prune(my_tree, test_mat)
    y_hat = create_forecast(my_tree, test_mat[:, 0:m - 1])
    error = y_hat - test_mat[:, m - 1]
    print("回归树等级预测准确率： %.1f%%." % (100 * (len(np.nonzero(abs(error) <= 0.6)[0]) / len(test_mat))))


test1('red_wine')
test2('red_wine')
