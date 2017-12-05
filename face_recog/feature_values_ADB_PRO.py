#print(__doc__)

# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

#from sklearn.externals.six.moves import zip(python3不需要这个我也不知道为什么)

import matplotlib.pyplot as plt
from sklearn import ensemble
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
#数据读取 前1953个数据行y值给出，用于训练模型，后351个数据y值未给出，用于预测
data= pd.read_csv('feature_values.csv')
X = data[['BL1','BL2','BL3','BL4','BL5','BL6','BR1','BR2','BR3','BR4','BR5','BR6',
          'EL1','EL2','EL3','EL4','EL5','EL6','ER1','ER2','ER3','ER4','ER5','ER6',
          'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','M1','M2','M3','M4',
          'M5','M6','M7','M8','M9','N2','N3','N4','N5','MFCC1','MFCC2','MFCC3',
          'MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','MFCC10','MFCC11',
          'MFCC12','E','Del1','Del2','Del3','Del4','Del5','Del6','Del7','Del8',
          'Del9','Del10','Del11','Del12','DelE','Acc1','Acc2','Acc3','Acc4',
          'Acc5','Acc6','Acc7','Acc8','Acc9','Acc10','Acc11','Acc12','AccE']]
y = data[['NAME']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

print(X_train.shape)
print(y_train.shape)

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=500,
    learning_rate=0.5)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=500,
    learning_rate=0.5,
    algorithm="SAMME")

bdt_real.fit(X_train, y_train)
bdt_discrete.fit(X_train, y_train)

real_test_errors = []
discrete_test_errors = []

for real_test_predict, discrete_train_predict in zip(bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
    real_test_errors.append(
        1. - accuracy_score(real_test_predict, y_test))
    discrete_test_errors.append(
        1. - accuracy_score(discrete_train_predict, y_test))

n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)
print(n_trees_discrete,n_trees_real)
# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

plt.figure(figsize=(15, 5))

plt.subplot(121)
plt.plot(range(1, n_trees_discrete + 1),
         discrete_test_errors, c='black', label='SAMME')
plt.plot(range(1, n_trees_real + 1),
         real_test_errors, c='black',
         linestyle='dashed', label='SAMME.R')
plt.legend()
#plt.ylim(0.18, 0.62)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')

plt.subplot(122)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
         "b", label='SAMME', alpha=.5)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
         "r", label='SAMME.R', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
         max(real_estimator_errors.max(),
             discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))
'''
plt.subplot(133)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
         "b", label='SAMME')
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))'''

# prevent overlapping y-axis labels
plt.subplots_adjust(wspace=0.25)
plt.show()