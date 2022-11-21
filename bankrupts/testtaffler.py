import numpy as np
from preprocessing_data import preprocess
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


a = [0.53, 0.13, 0.18, 0.16]  # 4ф Альтман

df_0 = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')
df_1 = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(2-3).csv')
df_2 = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(4-5).csv')
df_3 = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')
df_4 = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')

df = df_0.append(df_1, ignore_index=True).append(df_2, ignore_index=True).append(df_3, ignore_index=True) \
    .append(df_4, ignore_index=True)

x_train, x_test, y_train, y_test = train_test_split(df, df['Label'], test_size=0.4, random_state=42)

# bankrupts = df['bankrupt']
y = []
x = []

for index, row in x_test.iterrows():
    params = [row['2200 (2018)'] / row['1500 (2018)'],
              row['1200 (2018)'] / (row['1400 (2018)'] + row['1500 (2018)']),
              row['1500 (2018)'] / row['1600 (2018)'],
              row['2110 (2018)'] / row['1600 (2018)']]

    if np.all(np.isfinite(params)) and np.all(params != 0):
        x.append(params)
        y.append(row['Label'])


m = len(a)
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

    if z[i] <= 0.2:
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


print('TP: ', tp)
print('TN: ', tn)
print('FP: ', fp)
print('FN: ', fn)

print('Accuracy ', accuracy_score(temp_class, y))
print('Precision  ', precision_score(temp_class, y))
print('Recall ', recall_score(temp_class, y))
print('F1-score ', f1_score(temp_class, y))
print('Roc-Auc-score ', roc_auc_score(temp_class, y))

