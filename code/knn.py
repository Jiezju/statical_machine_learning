import math
import numpy as np
import matplotlib.pyplot as plt

def compute_dis(x, y, p=2):
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1 / p)
    else:
        return 0

class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 临近点个数
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.labels = {}

        for label in self.classes:
            self.labels[str(label)] = 0

    def predict(self, x):
        # 多节点
        if isinstance(x[0], list):
            test_results = []
            for pt in x:
                p_label = self.predict(pt)
                test_results.append(p_label)
            return test_results

        distance = {}
        for i in range(self.X_train.shape[0]):
            dis = compute_dis(self.X_train[i].tolist(), x)
            distance[str(i)] = (dis, self.y_train[i])

        # 取出n个点
        counts = sorted(distance.items(), key=lambda x: x[1][0])[:self.n] # list
        # print(counts)

        for k,v in counts:
            self.labels[str(v[1])] += 1

        p_label = '0'
        max_count = self.labels[p_label]
        for k, v in self.labels.items():
            if v > max_count:
                p_label = k
                max_count = v

        return int(p_label)

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)


if __name__ == '__main__':
    # dataset
    data = np.array([[5, 12, 1], [6, 21, 0], [14, 5, 0], [16, 10, 0], [13, 19, 0],
                     [13, 32, 1], [17, 27, 1], [18, 24, 1], [20, 20, 0], [23, 14, 1],
                     [23, 25, 1], [23, 31, 1], [26, 8, 0], [30, 17, 1],
                     [30, 26, 1], [34, 8, 0], [34, 19, 1], [37, 28, 1]])
    print(data.shape)

    # X, y
    train_x = data[:, :2]
    print(train_x.shape)
    train_y = data[:, 2]
    print(train_y.shape)

    # test case
    test_point = [16, 20]

    # plot dataset
    X0 = data[data[:, 2] == 0][:, :2] # [7, 2]
    X1 = data[data[:, 2] == 1][:, :2] # [11, 2]
    plt.scatter(X0[:, 0], X0[:, 1], label='0')
    plt.scatter(X1[:, 0], X1[:, 1], label='1')
    plt.plot(test_point[0], test_point[1], 'go', label='test_point')
    plt.xlabel('feature x_0')
    plt.ylabel('feature x_1')
    plt.legend()
    plt.show()

    # init knn
    knn = KNN(train_x, train_y)

    label = knn.predict([test_point])
    print(label)
