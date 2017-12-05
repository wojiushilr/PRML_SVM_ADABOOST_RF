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
y=[1,2,2,3,5,8,4,9,4,5,1,8,8,4,52,1,9,2,1,6,5,1,4,8]
print(y)