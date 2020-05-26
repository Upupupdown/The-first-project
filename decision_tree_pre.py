from create_tree import *


def decision_pre(filename):
    fr = open(filename)
    data_arr = []
    level = []
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
        level.append(float(cur_line[-1]))
        for i in range(len(cur_line)):
            line_arr.append(float(cur_line[i]))
        data_arr.append(line_arr)
    # 特征标签
    label = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    label1 = label.copy()
    m = int(len(data_arr) * 0.8)
    train_set = data_arr[:m]
    test_set = data_arr[m:]
    tree = create_tree(train_set, label1)
    error_count = 0.0
    for test in test_set:
        classify_result = classify(tree, label, test[:len(test) - 1])
        if classify_result is None:
            classify_result = majority_cnt(level)
        if classify_result != test[len(test) - 1]:
            error_count += 1.0
    print("决策树预测准确率： %.1f%%" % ((1 - error_count / len(test_set)) * 100))


decision_pre('red_wine')
