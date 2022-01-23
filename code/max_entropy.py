from copy import deepcopy
import math


class MaxEntropy(object):
    '''
        基于 GIS 的最大熵模型

    _samples:
        样本集
    _Y:
        标签种类集合
    _numXY:
        记录(x,y)出现次数的字典
    _N: 
        样本数
    _Ep_:
        经验估计的期望
    _xyID:
        记录(x,y)索引号的字典
    _n:
        不同(x,y)共现对的个数
    _C:
        样本中特征数最多的样本所含有的特征数
    _IDxy:
        与_xyID相反，通过索引找(x,y)
    _w:
        本轮训练的权重
    _epsilon:
        收敛条件
    _lastw:
        上一轮的权重
    '''

    def __init__(self, epsilon=0.01):
        self._samples = []
        self._Y = set()
        self._numXY = {}
        self._N = 0
        self._Ep_ = []
        self._xyID = {}
        self._n = 0
        self._C = 0
        self._IDxy = {}
        self._w = []
        self._epsilon = epsilon
        self._lastw = []

    def fit(self, dataset):
        self._samples = deepcopy(dataset)
        for items in self._samples:
            y = items[0]
            X = items[1:]
            self._Y.add(y)  # np.unique(y)
            for x in X:
                if (x, y) in self._numXY:
                    self._numXY[(x, y)] += 1
                else:
                    self._numXY[(x, y)] = 1

        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._C = max([len(sample) - 1 for sample in self._samples])
        self._w = [0] * self._n
        self._lastw = self._w[:]

        self._Ep_ = [0] * self._n
        for i, xy in enumerate(self._numXY):
            self._Ep_[i] = self._numXY[xy] / self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

    def _Zx(self, X):
        '''计算规范化因子 Z(x)'''
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            zx += math.exp(ss)
        return zx

    def _model_pyx(self, y, X):
        '''计算条件概率 p(y|x)'''
        zx = self._Zx(X)
        ss = 0
        for x in X:
            if (x, y) in self._numXY:
                ss += self._w[self._xyID[(x, y)]]
        pyx = math.exp(ss) / zx
        return pyx

    def _model_ep(self, index):
        '''计算模型期望'''
        x, y = self._IDxy[index]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self._model_pyx(y, sample)
            ep += pyx / self._N
        return ep

    def _convergence(self):
        '''判断是否全部收敛'''
        for last, now in zip(self._lastw, self._w):
            if abs(last - now) >= self._epsilon:
                return False
        return True

    def predict(self, X):
        Z = self._Zx(X)
        result = {}
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            pyx = math.exp(ss) / Z
            result[y] = pyx
        return result

    def train(self, maxiter=400):
        for loop in range(maxiter):
            self._lastw = self._w[:]
            for i in range(self._n):
                ep = self._model_ep(i)
                self._w[i] += math.log(self._Ep_[i] / ep) / self._C
            if self._convergence():
                break
        print('Done')


if __name__ == '__main__':
    dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
               ['no', 'sunny', 'hot', 'high', 'TRUE'],
               ['yes', 'overcast', 'hot', 'high', 'FALSE'],
               ['yes', 'rainy', 'mild', 'high', 'FALSE'],
               ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
               ['no', 'rainy', 'cool', 'normal', 'TRUE'],
               ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
               ['no', 'sunny', 'mild', 'high', 'FALSE'],
               ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
               ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
               ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
               ['yes', 'overcast', 'mild', 'high', 'TRUE'],
               ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
               ['no', 'rainy', 'mild', 'high', 'TRUE']]

    maxent = MaxEntropy()
    x = ['overcast', 'mild', 'high', 'FALSE']

    maxent.fit(dataset)
    maxent.train()
    print('predict:', maxent.predict(x))
