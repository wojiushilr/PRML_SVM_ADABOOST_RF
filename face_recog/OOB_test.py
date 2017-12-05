import numpy as np
import pandas as pd
import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)

from tabulate import tabulate
from collections import Counter

rd.seed(71)
n_data = 100
result = np.zeros((12, n_data))

OOBs = []

plt.figure(figsize=(16,8))
for i in range(12):
    ax = plt.subplot(4,3,i+1)
    result[i] = rd.randint(1,n_data+1, n_data)
    res = plt.hist(result[i], bins=range(1,n_data+1))
    cnt = Counter(res[0])
    plt.title("# of OOB = {}".format(cnt[0]))
    OOBs.append(cnt[0])
    plt.xlim(0, n_data)
plt.tight_layout()
print("Average of OOB = {}".format(np.mean(OOBs)))
plt.show()
#print( rd.randint(1,10+1,5)) 5指的是取样次数