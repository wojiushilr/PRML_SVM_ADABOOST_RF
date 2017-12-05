#svm
from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics

from sklearn.linear_model import LinearRegression
data= pd.read_csv('data3.csv')
data1=data[:1953]
data2 = data[1953:]
#print(data1.shape)
X_train = data1[['stadium','home_score','away_score','weather','temperature','humidity']]
y_train= data1['y']
X_train_n = np.array(X_train)
y_train_n= np.array(y_train)

X_test = data2[['stadium','home_score','away_score','weather','temperature','humidity']]
X_test_n = np.array(X_test)
#print(type(y_train))
#print(X_train_n)


svr_rbf = SVR(kernel='rbf', C=1e5, gamma=0.1)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=3)
y_rbf = svr_rbf.fit(X_train_n, y_train_n.ravel())
#y_lin = svr_lin.fit(X_train_n, y_train_n.ravel()).predict(X_train_n)
#y_poly = svr_poly.fit(X_train_n, y_train_n.ravel()).predict(X_train_n)
###
test_rbf = svr_rbf.predict(X_train_n)
#test_lin = svr_lin.predict(X_test_n)
#test_poly = svr_poly.predict(X_test_n)

print(test_rbf)
print(y_train_n)
print(y_rbf.intercept_)
# 相関係数計算
rbf_corr = np.corrcoef(y_train_n, test_rbf)[0, 1]

# RMSEを計算
rbf_rmse = np.sqrt(metrics.mean_squared_error(y_train_n, test_rbf))


print ("svr: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr))
#np.savetxt('new.csv', test_rbf, delimiter = ',')

i=[]
for k in range(len(X_train_n)):
    i.append(k)
j=[]
for n in range(len(y_train)):
    j.append(n)
fig = plt.figure(figsize=(7,4))
plt.figure(1)
plt.plot(i,test_rbf)
plt.figure(2)
plt.plot(j,y_train)
plt.show()




'''
rbf1=svm.SVR(kernel='rbf',C=1, )#degree=2,,gamma=, coef0=
rbf2=svm.SVR(kernel='rbf',C=20, )#degree=2,,gamma=, coef0=
poly=svm.SVR(kernel='poly',C=1,degree=2)
rbf1.fit(X_train_n,y_train_n)
rbf2.fit(X_train_n,y_train_n)
result1 = rbf1.predict(X_test_n)
result2 = rbf2.predict(X_test_n)
print(result2)
'''


