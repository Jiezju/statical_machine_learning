import math
import numpy as np

class NaiveBayes:
    def __init__(self, X, y):
        self.train_x = X
        self.train_y = y
        self.label = np.unique(self.train_y).tolist()
        self.num_features = self.train_x.shape[1]


        # compute prior_probability  and conditional_probability

        # 1. compute prior_probability 类别概率
        self.prior_probability = np.zeros(shape=(len(self.label), ), dtype=np.float32)

        for i,label in enumerate(self.label):
            label_counts = self.train_y[self.train_y == label].shape[0]
            self.prior_probability[i] = label_counts / self.train_y.shape[0]

        # 2. conditional_probability
        self.conditional_probability = {}

        # 遍历每个特征 第 i 个特征
        for i in range(self.num_features):
            feature = self.train_x[:, i]
            feature_val = np.unique(feature)
            self.conditional_probability[str(i)] = {}
            f_par = self.conditional_probability[str(i)]
            for fval in feature_val:
                f_par[fval] = {}
                for label in self.label:
                    fval_x = feature[self.train_y == label]
                    f_par[fval][label] = fval_x[fval_x == fval].shape[0] / self.train_y[self.train_y == label].shape[0]

    def predict(self, x):
        results_prob = {}
        for label in self.label:
            prod = self.prior_probability[self.label.index(label)]
            for i, feat in enumerate(x):
                prod *= self.conditional_probability[str(i)][str(feat)][label]
            results_prob[label] = prod

        label = self.label[0]
        prob = results_prob[label]
        for k,v in results_prob.items():
            if v > prob:
                label = k
                prob = v

        return label


if __name__ == '__main__':
    # dataset
    data = np.array([[1, 'S', -1], [1, 'M', -1], [1, 'M', 1], [1, 'S', 1], [1, 'S', -1],
                     [2, 'S', -1], [2, 'M', -1], [2, 'M', 1], [2, 'L', 1], [2, 'L', 1],
                     [3, 'L', 1], [3, 'M', 1], [3, 'M', 1], [3, 'M', 1],
                     [3, 'L', 1], [3, 'L', -1]])

    train_x = data[:, :2]
    train_y = data[:, 2]
    nbayes = NaiveBayes(train_x, train_y)

    x = [2, 'S']
    label = nbayes.predict(x)
    print(label)


