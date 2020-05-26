from math import log
import operator


def cal_ent(data_set):
    """
    计算给定数据集的香农熵
    :param data_set: 数据集
    :return: 香农熵
    """
    data_num = len(data_set)
    label_counts = {}
    # 为所有可能分类创建字典，记录概数
    for feat_vec in data_set:
        cur_label = feat_vec[-1]
        if cur_label not in label_counts.keys():
            label_counts[cur_label] = 0
        label_counts[cur_label] += 1
    ent = 0.0
    # 计算香农熵
    for key in label_counts:
        prob = float(label_counts[key]) / data_num
        ent -= prob * log(prob, 2)
    return ent


def split_data_set(data_set, axis, value):
    """
    按照给定特征划分数据集
    :param data_set: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需返回的特征的值
    :return:
    """
    # 为了不修改原始数据集，创建一个新的列表对象
    ret_data_set = []
    for feat_vec in data_set:
        # 将符合特征的数据抽取出来
        if feat_vec[axis] == value:
            # 抽取按某一特征划分后的数据集部分，得到某一特征数据中的其他特征与类别以计算划分后的香农熵
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature(data_set):
    # 数据集中特征数
    feature_num = len(data_set[0]) - 1
    # 计算并保存基本香农熵（即未划分前的）
    base_ent = cal_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    # 循环按不同的特征划分并计算划分后的香农熵以得到信息增值
    for i in range(feature_num):
        # 得到某一特征所有可能包含的值
        feat_list = [example[i] for example in data_set]
        unique_val = set(feat_list)
        new_ent = 0.0
        for value in unique_val:
            sub_data_set = split_data_set(data_set, i, value)
            # 计算根据第i个特征的某个特征值划分集合到一起的小集合的概率值
            prob = len(sub_data_set) / float(len(data_set))
            # 计算根据第i个特征划分后的香农熵，即各个小集合和熵之和
            new_ent += prob * cal_ent(sub_data_set)
        # 计算信息增值
        info_gain = base_ent - new_ent
        # 某种划分方式获得的信息增值越大，混乱度降低越多，即该种划分方式越好，则替换记录
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """
    找出出现次数最多的类别以作为无法确定类别的归属
    :param class_list: 类别列表
    :return: 出现次数最高的类别
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    创建决策树
    :param data_set: 数据集
    :param labels: 标签集
    :return: 创建好的树
    """
    # 存储数据集中的类别
    class_list = [example[-1] for example in data_set]
    # 若当前数据集中只含有一种类型，即第一种类型的数量等于数据集中类型数，则已分好该类，返回类标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 若当前数据集中只有类型结果，没有特征，则无法正确返回类标签，将次数出现最多的类别作为类标签
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    # 选择划分方式最好的方式
    best_feat = choose_best_feature(data_set)
    best_feat_label = labels[best_feat]
    # print(f'({best_feat_label})', end=' ')
    my_tree = {best_feat_label: {}}
    # 删除已利用其划分过的类特征
    del (labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_val = set(feat_values)
    for value in unique_val:
        sub_labels = labels[:]
        # 递归以下一个特征值作为划分依据
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树执行分类
    :param input_tree: 决策树
    :param feat_labels: 标签集
    :param test_vec: 测试集
    :return: 下一层决策树或返回叶子节点所属类别
    """
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    # 记录决策判断的下标已判断输入相应的特征
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                # 递归进入决策树下一层
                return classify(second_dict[key], feat_labels, test_vec)
            else:
                # 已判断至叶子结点，说明找到测试点所属类别
                return second_dict[key]


