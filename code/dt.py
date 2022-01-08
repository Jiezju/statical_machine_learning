import math

class ID3Tree:
    def __init__(self, dataset, feature_labels):
        self.dataset = dataset
        self.feature_labels = feature_labels
        self.tree = self.build_tree(self.dataset, self.feature_labels)
    
    # 计算给定特征取值的经验熵 H(D)
    def calc_ent(self, dataset):
        n = len(dataset)
        label_counts = {}
        # 统计每个类别出现的次数
        for feature in dataset:
            label = feature[-1]
            if label not in label_counts:
                label_counts[label] = 0     # 创建该元素并清零
            label_counts[label] += 1

        # 计算熵
        entropy = 0
        for key in label_counts:
            p = float(label_counts[key]) / n  # 计算类概率，或者说类在所有数据中的比例
            entropy -= p * math.log(p, 2)
        return entropy

    # 计算给定特征的经验条件熵 H(D | A), axis 表示第 axis 列特征
    def cond_ent(self, dataset, axis):
        data_length = len(dataset)
        feature_sets = {}

        for i in range(data_length):
            feature = dataset[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(dataset[i])

        cond_ent = 0
        for feature in feature_sets:
            p = len(feature_sets[feature]) / len(dataset)
            if p > 1:
                print(0)
            cond_ent += p * self.calc_ent(feature_sets[feature])
        return cond_ent

    # 信息增益 g(D,A)
    def info_gain(self, ent, cond_ent):
        return ent - cond_ent

    # 划分数据集 去除指定特征属性值的剩余数据集
    def split_dataset(self, dataSet, axis, value):
        rdataset = []
        for feat_vec in dataSet:
            if feat_vec[axis] == value:
                # 去除 axis 特征
                new_feat_vec = feat_vec[:axis] + feat_vec[axis+1:]
                rdataset.append(new_feat_vec)
        return rdataset

    # 依据最优的信息增益获取数据最优划分方式
    def choose_best_feature_to_split(self, dataSet):
        num_features = len(dataSet[0]) - 1
        h_d = self.calc_ent(dataSet)
        best_feature = -1
        best_info_gain = 0.0
        for i in range(num_features):
            ifgain = self.info_gain(h_d, self.cond_ent(dataSet, axis=i))
            if ifgain > best_info_gain:
                best_info_gain = ifgain
                best_feature = i

        return best_feature

    # 众数表决器
    def majority_cnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(),
                                key=lambda x: x[1], reverse=True)
        return sortedClassCount[0][0]
    
    # ID3 构建决策树 f_name 特征名称
    def build_tree(self, dataSet, f_name):
        class_list = [instance[-1] for instance in dataSet]

        # 如果只有一个类别，返回
        if len(set(class_list)) == 1:
            return class_list[0]

        # 如果所有特征都被遍历完了，返回出现次数最多的类别
        if len(dataSet[0]) == 1:
            return self.majority_cnt(class_list)

        best_feat = self.choose_best_feature_to_split(dataSet)
        best_feat_name = f_name[best_feat]
        tree = {best_feat_name: {}}
        del f_name[best_feat]

        # 获取子节点， 特征取值个数
        feat_values = [instance[best_feat] for instance in dataSet]
        unique_values = set(feat_values)
        for value in unique_values:
            tree[best_feat_name][value] = self.build_tree(self.split_dataset(dataSet, best_feat, value), f_name[:])

        return tree
    
    # 预测
    def predict(self, x):
        pass


if __name__ == '__main__':
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']

    tree = ID3Tree(datasets, labels)


