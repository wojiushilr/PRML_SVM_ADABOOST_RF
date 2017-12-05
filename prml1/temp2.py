import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# インプットを適当に生成
X1 = np.sort(5 * np.random.rand(40, 1).reshape(40), axis=0)
X2 = np.sort(3 * np.random.rand(40, 1).reshape(40), axis=0)
X3 = np.sort(9 * np.random.rand(40, 1).reshape(40), axis=0)
X4 = np.sort(4 * np.random.rand(40, 1).reshape(40), axis=0)

# インプットの配列を一つに統合
X = np.c_[X1, X2, X3, X4]


# アウトプットを算出
y = np.sin(X1).ravel() + np.cos(X2).ravel() + np.sin(X3).ravel() - np.cos(X4).ravel()

y_o = y.copy()
print(type(y_o))
print(type(y))

# ノイズを加える
y[::5] += 3 * (0.5 - np.random.rand(8))

# フィッティング
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=3)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# テストデータも準備
test_X1 = np.sort(5 * np.random.rand(40, 1).reshape(40), axis=0)
test_X2 = np.sort(3 * np.random.rand(40, 1).reshape(40), axis=0)
test_X3 = np.sort(9 * np.random.rand(40, 1).reshape(40), axis=0)
test_X4 = np.sort(4 * np.random.rand(40, 1).reshape(40), axis=0)

test_X = np.c_[test_X1, test_X2, test_X3, test_X4]
test_y = np.sin(test_X1).ravel() + np.cos(test_X2).ravel() + np.sin(test_X3).ravel() - np.cos(test_X4).ravel()

# テストデータを突っ込んで推定してみる
test_rbf = svr_rbf.predict(test_X)
test_lin = svr_lin.predict(test_X)
test_poly = svr_poly.predict(test_X)

print(test_rbf[:2])
from sklearn.metrics import mean_squared_error
from math import sqrt
'''
# 相関係数計算
rbf_corr = np.corrcoef(test_y, test_rbf)[0, 1]
lin_corr = np.corrcoef(test_y, test_lin)[0, 1]
poly_corr = np.corrcoef(test_y, test_poly)[0, 1]

# RMSEを計算
rbf_rmse = sqrt(mean_squared_error(test_y, test_rbf))
lin_rmse = sqrt(mean_squared_error(test_y, test_lin))
poly_rmse = sqrt(mean_squared_error(test_y, test_poly))

print ("RBF: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr))
print ("Linear: RMSE %f \t Corr %f" % (lin_rmse, lin_corr))
print ("Poly: RMSE %f \t\t Corr %f" % (poly_rmse, poly_corr))
'''