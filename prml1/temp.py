from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
'''
X = np.arange(-5.,9.,0.1)
X=np.random.permutation(X)
X_=[[i] for i in X]
print (type(X_))
b=5.
y=0.5 * X ** 2.0 +3. * X + b + np.random.random(X.shape)* 10.
y_=[i for i in y]'''
'''
rbf1=svm.SVR(kernel='rbf',C=1, )#degree=2,,gamma=, coef0=
rbf2=svm.SVR(kernel='rbf',C=20, )#degree=2,,gamma=, coef0=
poly=svm.SVR(kernel='poly',C=1,degree=2)
rbf1.fit(X_,y_)
rbf2.fit(X_,y_)
poly.fit(X_,y_)
result1 = rbf1.predict(X_)
result2 = rbf2.predict(X_)
result3 = poly.predict(X_)
#plt.hold(True)
plt.plot(X,y,'bo',fillstyle='none')
plt.plot(X,result1,'r.')
plt.plot(X,result2,'g.')
plt.plot(X,result3,'c.')
plt.show()'''

from sklearn.svm import SVR
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
print(type(y))
clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X, y)
result = clf.predict(X)
plt.plot(X,y,'bo',fillstyle='none')
plt.plot(X,result,'r.')
plt.show()