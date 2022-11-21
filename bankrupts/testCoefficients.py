import numpy as np
from preprocessing_data import preprocess
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import math
import pandas as pd


# a = [7.99078972e+00, 5.04799395e+00, 1.04080848e+01, -6.32767275e-03]
# a = [6.56, 3.26, 6.72, 1.05]  # Альтман
a = [0.02796103, 15.99467552, -14.49268392, -11.0628872]

# fileName = '../data/financeData.csv'
# df = pd.read_csv(fileName)
df = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(2-3).csv')

m = 4

# bankrupts = df['bankrupt']
y = []
x = []

for index, row in df.iterrows():
    params = [(row['1200 (2018)'] - row['1500 (2018)']) / row['1600 (2018)'], row['1370 (2018)'] / row['1600 (2018)'],
              row['2300 (2018)'] / row['1600 (2018)'], row['1300 (2018)'] / (row['1400 (2018)'] + row['1500 (2018)'])]

    """for j in range(m):
        if (params[j] is None) or not (params[j] >= 0 or params[j] <= 0):
            params[j] = 0
        elif np.abs(params[j] > 100000):
            params[j] /= 1000"""

    if np.all(np.isfinite(params)) and np.all(params != 0):
        x.append(params)
        y.append(row['Label'])


n = len(y)
v = 0
z = [0 for _ in range(n)]
answers = []
temp_class = []
tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(x)):
    z[i] = 0
    for j in range(m):
        z[i] += a[j] * x[i][j]

    if z[i] > 1:
        temp_class.append(1)
        if temp_class[i] == y[i]:
            tp += 1
        else:
            fp += 1
    else:
        temp_class.append(0)
        if temp_class[i] == y[i]:
            tn += 1
        else:
            fn += 1

    """if z[i] > 0.5:
        temp_class.append(1)
        if temp_class[i] == y[i]:
            tp += 1
        else:
            fp += 1
    else:
        temp_class.append(0)
        if temp_class[i] == y[i]:
            tn += 1
        else:
            fn += 1"""

    """if z[i] >= 0:
        z[i] = 1
        print(i+2, ' - Является банкротом')
    else:
        z[i] = 0
        print(i+2, ' - Не является банкротом')"""

print('TP: ', tp/n)
print('TN: ', tn/n)
print('FP: ', fp/n)
print('FN: ', fn/n)

print('Accuracy ', accuracy_score(temp_class, y))
print('Precision  ', precision_score(temp_class, y))
print('Recall ', recall_score(temp_class, y))
print('F1-score ', f1_score(temp_class, y))
print('Roc-Auc-score ', roc_auc_score(temp_class, y))

