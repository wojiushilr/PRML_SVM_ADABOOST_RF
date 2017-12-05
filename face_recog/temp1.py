import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
X1, y1 = make_gaussian_quantiles(cov=2.0,n_samples=500, n_features=2,n_classes=2, random_state=1)
# 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为2
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,n_samples=400, n_features=2, n_classes=2, random_state=1)
#讲两组数据合成一组数据
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))

print('X1',len(X1))
print('y1',len(y1))
print('X2',len(X2))
print('y2',len(y2))
print(len(X[:, 0]))
print(len(X[:, 1]))
print(y1.shape)
print(X1[0:5])
print(X2[0:5])
print(X[0, 0]) #X数据是900x2的矩阵，这个打印的是第1行第1列的数字
print(X[0, 1]) #这个打印的是第1行第二列的数字
print(X[0, 3])

#print(X2)
#print(y1)
#print(y2)