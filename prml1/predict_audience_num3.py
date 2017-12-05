#adaboost 随机森林
from sklearn.svm import SVR
from sklearn import ensemble
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
#数据读取 前1953个数据行y值给出，用于训练模型，后351个数据y值未给出，用于预测
data= pd.read_csv('data3.csv')
data1=data[:1953]
data2 = data[1953:]
#print(data1.shape)
#数据自变量选用10个参数
X_train = data1[['stage','match','gameday','time','stadium','home_score','away_score','weather','temperature','humidity']]
y_train= data1['y']
X_train_n = np.array(X_train)
y_train_n= np.array(y_train)
X_test = data2[['stage','match','gameday','time','stadium','home_score','away_score','weather','temperature','humidity']]
X_test_n = np.array(X_test)
#print(type(y_train))
print(X_test_n)
print(X_train_n)

#对象方法实例
rf =ensemble.RandomForestRegressor(n_estimators=50)#这里使用20个决策树
ada = ensemble.AdaBoostRegressor(n_estimators=50)
#开始拟合
y_rbf = ada.fit(X_train_n, y_train_n.ravel())
y_rf= rf.fit(X_train_n, y_train_n.ravel())
#y_lin = svr_lin.fit(X_train_n, y_train_n.ravel()).predict(X_train_n)
#y_poly = svr_poly.fit(X_train_n, y_train_n.ravel()).predict(X_train_n)
#开始预测
test_ada = ada.predict(X_train_n)
test_rf = rf.predict(X_train_n)
#test_lin = svr_lin.predict(X_test_n)
#test_poly = svr_poly.predict(X_test_n)
#print(test_rbf)
#数据输出
#np.savetxt('new_rf.csv', test_rf, delimiter = ',')

#模型评估
# 相関係数計算
ada_corr = np.corrcoef(y_train_n, test_ada)[0, 1]
rf_corr = np.corrcoef(y_train_n,test_rf)[0, 1]
# RMSEを計算
ada_rmse = np.sqrt(metrics.mean_squared_error(y_train_n, test_ada))
rf_rmse = np.sqrt(metrics.mean_squared_error(y_train_n, test_rf))
print ("ada: RMSE %f \t\t Corr %f" % (ada_rmse, ada_corr))
print ("rf: RMSE %f \t\t Corr %f" % (rf_rmse, rf_corr))

#结果可视化
i=[]
for k in range(len(X_train_n)):
    i.append(k)
j=[]
for n in range(len(y_train)):
    j.append(n)
fig = plt.figure(figsize=(7,4))
plt.figure(1)
plt.plot(i,test_ada)
plt.figure(2)
plt.plot(j,y_train)
plt.show()



