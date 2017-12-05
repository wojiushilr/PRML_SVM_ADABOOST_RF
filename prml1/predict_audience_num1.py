#Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression

data= pd.read_csv('data3.csv')
data1=data[:1953]
#print(data1.shape)
X_train = data1[['stage','match','gameday','time','stadium','home_score','away_score','weather','temperature','humidity']]
y_train= data1['y']
#X_test =
#print(X.head)
#print(y_train)

data2 = data[1953:]
#print(data2.shape)
X_test = data2[['stage','match','gameday','time','stadium','home_score','away_score','weather','temperature','humidity']]
print(data2['stage'].shape)
dates = pd.date_range('20120310',periods=873)#dateIndex
#print(dates)

linreg = LinearRegression()
linreg.fit(X_train,y_train)
print(linreg.intercept_)
print(linreg.coef_)

y_pred = linreg.predict(X_train)
#print(y_pred)
#np.savetxt('o.csv',y_pred,delimiter=',')
#save = pd.DataFrame({'yp':[y_pred]})
#save.to_csv('o.txt',sep=',')
# 相関係数計算
rbf_corr = np.corrcoef(y_train, y_pred)[0, 1]

# RMSEを計算
rbf_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_pred))


print ("LR: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr))


'''
rbf1 = svm.SVR(kernel='rbf',C=1,)
rbf2 = svm.SVR(kernel='rbf',C=20,)
rbf1.fit(X,y)
rbf2.fit(X,y)

result1= rbf1.predict(X)
result2= rbf2.predict(X)

plt.plot(X,y,'bo',fillstyle='none')
plt.plot(X,result1,'r.')
plt.plot(X,result2,'g.')

plt.show()
'''

'''
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
plt.sca(ax)

plt.plot(data['id'],data['y'])
plt.show()
'''