#adaboost 随机森林
from sklearn.svm import SVR
from sklearn import ensemble
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
#random_stat=1是每次生成的测试训练集都是相同的，为0时每次生成不同
print(X_train.shape)
print(X_test.shape)
#print(y_train.head())
#print(type(y_test))
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
#对象方法实例
#ada = ensemble.AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
test_err_rf = []
test_err_ada =[]

for i in range(300):

    rf = ensemble.RandomForestClassifier(n_estimators=i+1)
    ada = ensemble. AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=i+1,
    learning_rate=0.5)
    #开始拟合
    rf.fit(X_train, y_train.ravel())
    ada.fit(X_train, y_train.ravel())
    #开始预测
    #y_pre = ada.predict(X_test)
    #print('y_pre:',y_pre)
    #print('y_test;',y_test)
    #accuracy example1: high variance estimate高方差估计
    if i %50==0:

       print('rf:',metrics.accuracy_score(y_test,rf.predict(X_test)))
       print('ada',metrics.accuracy_score(y_test,ada.predict(X_test)))

    test_err_rf.append(1-(metrics.accuracy_score(y_test,rf.predict(X_test))))
    test_err_ada.append(1-(metrics.accuracy_score(y_test,ada.predict(X_test))))


plt.figure(figsize=(15, 5))
plt.plot(range(1, 300 + 1),
         test_err_rf, c='black',)
plt.plot(range(1, 300 + 1),
         test_err_ada, c='red')
plt.legend(["RandomForest",'Adaboost'])
#plt.ylim(0.18, 0.62)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')


plt.show()







'''
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


'''
