'''
perceptron 对偶形式实现

define：

    X = [x_1,x_2,...,x_n]
    y = [y_1,y_2,...,y_n]
'''

import numpy as np

def perceptron(X, y):
    eta = 1

    m, n = X.shape # m 特征数量， n 样本数量

    alpha = np.zeros((n,))
    b = 0

    G = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            G[i,j] = np.dot(X[:,i].T, X[:,j])

    while True:
        count = 0
        for i in range(n):
            _sum_w = 0
            for j in range(n):
                _sum_w += alpha[j]*y[j]*G[j,i]
            
            if y[i]*(_sum_w + b) <=0:
                alpha[i] = alpha[i] + eta
                b = b + eta * y[i]

                count += 1
        if count == 0:
            break
    
    return alpha,b

def perceptron_predict(w, b, x):
    y = np.dot(w.T, x) + b
    if y < 0:
        return -1
    else: 
        return 1


if __name__ == '__main__':
    x_1 = [3,3]
    x_2 = [4,3]
    x_3 = [1,1]
    X = np.array([x_1,x_2,x_3]).T
    y = np.array([1,1,-1])
    alpha, b = perceptron(X, y)

    w = 0
    for i in range(X.shape[1]):
        w += alpha[i] * y[i] * X[:,i] + b
    y = perceptron_predict(w, b, np.array(x_3))

