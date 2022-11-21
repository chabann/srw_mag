from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from scoring import Scoring
from preprocessing_data import preprocess
from preprocess2 import preprocess2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]  # deleting the column from the dataset

    # print(dataset)
    return dataset


df_0 = preprocess2('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')
df_1 = preprocess2('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(2-3).csv')
df_2 = preprocess2('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(4-5).csv')
df_3 = preprocess2('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')
df_4 = preprocess2('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')

df = df_0.append(df_1, ignore_index=True).append(df_2, ignore_index=True).append(df_3, ignore_index=True) \
    .append(df_4, ignore_index=True)

# df = correlation(df, 0.7)
df1 = df.drop('Label', axis=1)

x_train, x_test1, y_train, y_test1 = train_test_split(df1, df['Label'], test_size=0.2, random_state=42)

x_test, x_valid, y_test, y_valid = train_test_split(x_test1, y_test1, test_size=0.2, random_state=42)

print('В тренировочной выборке компаний-банкротов: ', y_train[y_train == 1].shape[0])
print('В тренировочной выборке не банкротов: ', y_train[y_train == 0].shape[0])
print('')

y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)
y_valid_cat = to_categorical(y_valid, 2)


model = Sequential()
model.add(Dense(150, activation='relu', input_dim=x_test.shape[1]))
model.add(Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)))
model.add(Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='sgd',
              metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.summary()

hist = model.fit(x_train, y_train_cat, validation_data=(x_valid, y_valid_cat), epochs=10, batch_size=100)


datas = [(x_test, y_test)]
names = ['тестовых']

for i in range(len(datas)):
    predicted = model.predict(datas[i][0])
    predicted_val = []
    predicted_prob = []

    for j in range(len(predicted)):
        if predicted[j][0] > predicted[j][1]:
            predicted_val.append(0)
        else:
            predicted_val.append(1)

        predicted_prob.append(predicted[j][1])

    # scoring = Scoring(predicted_val, datas[i][1])
    # print('RMSE для ' + names[i] + ' данных ', scoring.rmse())

    scoring = Scoring(predicted_val, datas[i][1], True)
    print('Precision для ' + names[i] + ' данных ', scoring.precision())
    print('Accuracy для ' + names[i] + ' данных ', scoring.accuracy())
    print('Recall для ' + names[i] + ' данных ', scoring.recall())
    print('F1-score для ' + names[i] + ' данных ', scoring.f1_score())
    #print('')
    #print('Accuracy для ' + names[i] + ' данных ', accuracy_score(predicted_val, datas[i][1]))
    #print('Precision для ' + names[i] + ' данных ', precision_score(predicted_val, datas[i][1]))
    #print('Recall для ' + names[i] + ' данных ', recall_score(predicted_val, datas[i][1]))
    #print('F1-score для ' + names[i] + ' данных ', f1_score(predicted_val, datas[i][1]))
    #print('Roc-Auc-score для ' + names[i] + ' данных ', roc_auc_score(predicted_val, datas[i][1]))
    print('')

