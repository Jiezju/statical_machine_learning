from email import iterators
import math
import numpy as np


class LogisticModel:
    def __init__(self, X, Y, learning_rate=0.001, iterations=100):
        self.x = self.data_preprocessing(X)
        self.y = Y
        self.learning_rate = learning_rate
        self.iterations = iterations
        self._w = self._init_param()

    def data_preprocessing(self, x):
        ones = np.ones(shape=(10, 1), dtype=np.float)
        newX = np.concatenate([x, ones], axis=-1)
        return newX

    def _init_param(self):
        num_features = self.x.shape[1]
        limit = 1 / math.sqrt(num_features)
        return np.random.uniform(-limit, limit, (num_features + 1,))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(np.dot(self._w, x)))

    def train(self):
        print("Start training...")
        for i in range(self.iterations):
            sum = 0
            for x, y in zip(self.x, self.y):
                sum += (x*y - x * np.exp(np.dot(self._w, x) * self._sigmoid(x)))

            self._w -= self.learning_rate * sum

        print("Finish training !")

    def predict(self, x):
        p_0 = self._sigmoid(x)
        p_1 = np.exp(np.dot(self._w, x)) * self._sigmoid(x)
        
        return 0 if p_0 > p_1 else 1



if __name__ == '__main__':
    # 生成随机数据集
    X=np.random.normal(size=(10, 5))
    Y=np.random.randint(low=0, high=2, size=(10, 1))

    lm=LogisticModel(X, Y)
    print(X.shape)
    print(Y.shape)
