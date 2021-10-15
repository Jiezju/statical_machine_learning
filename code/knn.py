'''
KNN
'''

import numpy as np

def euro_dis(x, y):
    if x.shape[0] != y.shape[0]:
        exit(-1)

    dis = 0
    for i in range(x.shape[0]):
        dis += np.sqrt((x[i] * x[i]) + (y[i]*y[i]))
    
    return dis

class KNN:
    def __init__(self, dis, X, y, k):
        self._disFunc = dis
        self._datas = X # n*m
        self._labels = y # m*1
        self._K = k
        self._distance = [0] * self._datas.shape[1]
        self._class = [0] *  self._datas.shape[1] # 多数表决器

    def _patition(self, l, r, pivot):
        i = l+1
        j = r
        while i <= j:
            while i<=r and self._distance[i] < pivot:
                i += 1
            while j >= l+1 and self._distance[j] > pivot:
                j -= 1
            
            if i <= j:
                self._class[i], self._class[j] = self._class[j], self._class[i]
                self._distance[i], self._distance[j] = self._distance[j], self._distance[i]
                i += 1
                j -= 1
        self._class[j], self._class[l] = self._class[l], self._class[j]
        self._distance[j], self._distance[l] = self._distance[l], self._distance[j]

        return j
    
    def _sort_helper(self, l, r):
        if l >= r:
            return
        pivot = self._distance[l]
        index = self._patition(l, r, pivot)

        self._sort_helper(l, index-1)
        self._sort_helper(index+1, r)

    def _sort(self):
        self._sort_helper(0, self._datas.shape[1] - 1)

    def predict(self, x):
        for i in range(self._datas.shape[1]):
            x_i = self._datas[:,i]
            self._distance[i] = self._disFunc(x, x_i)
            self._class[i] = self._labels[i]
        
        self._sort()

        result_dic = {}

        for ele in self._class:
            if ele not in result_dic:
                result_dic[ele] = 1
            else:
                result_dic[ele] += 1

        max_count = -1
        predict_label = None
        for k, v in result_dic:
            if v > max_count:
                predict_label = k
                max_count = v

        return predict_label