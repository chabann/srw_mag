from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing_data import preprocess
from preprocess2 import preprocess2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from scoring import Scoring
import numpy as np
import pickle


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


df_0 = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')
df_1 = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(2-3).csv')
df_2 = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(4-5).csv')
df_3 = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')
df_4 = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')

df = df_0.append(df_1, ignore_index=True).append(df_2, ignore_index=True).append(df_3, ignore_index=True) \
    .append(df_4, ignore_index=True)

df = correlation(df, 0.7)
df1 = df.drop('Label', axis=1)

# x_use, x_no_use, y_use, y_no_use = train_test_split(df1, df['Label'], test_size=0.3, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(df1, df['Label'], test_size=0.2, random_state=42)

print('В тренировочной выборке компаний-банкротов: ', y_train[y_train == 1].shape[0])
print('В тренировочной выборке не банкротов: ', y_train[y_train == 0].shape[0])
print('')

print('is None ', np.any(np.isnan(x_train)))
print('is not Inf ', np.all(np.isfinite(x_train)))
"""
param_grid = {
    'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
    'max_iter': [50, 100, 150],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

mlp = MLPClassifier(hidden_layer_sizes=(150, 100, 50),
                    max_iter=300, activation='relu',
                    solver='adam')

grid = GridSearchCV(mlp, param_grid, n_jobs= -1, cv=5)
grid.fit(x_train, y_train)

print(grid.best_params_)

grid_predictions = grid.predict(x_test)

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, grid_predictions)))

"""
mlp = MLPClassifier(hidden_layer_sizes=(150, 100, 50),
                    max_iter=500, activation='tanh', alpha=0.0001, learning_rate='adaptive',
                    solver='adam')

pipe = Pipeline([('scaler', StandardScaler()), ('mlpc', mlp)])

pipe.fit(x_train, y_train)
print(pipe.score(x_test, y_test))  # get accuracy

datas = [(x_train, y_train), (x_test, y_test)]
names = ['тренировочных', 'тестовых']

for i in range(len(datas)):
    predicted_val = pipe.predict(datas[i][0])

    # scoring = Scoring(predicted_val, datas[i][1])
    # print('RMSE для ' + names[i] + ' данных ', scoring.rmse())

    scoring = Scoring(predicted_val, datas[i][1], True)
    print('Precision для ' + names[i] + ' данных ', scoring.precision())
    print('Accuracy для ' + names[i] + ' данных ', scoring.accuracy())
    print('Recall для ' + names[i] + ' данных ', scoring.recall())
    print('F1-score для ' + names[i] + ' данных ', scoring.f1_score())
    print('AUC-score для ' + names[i] + ' данных ', roc_auc_score(predicted_val, datas[i][1]))
    print('')

filename = '../data/models/short_nn.sav'
pickle.dump(pipe, open(filename, 'wb'))


"""
{'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (150, 100, 50), 'learning_rate': 'adaptive', 'max_iter': 150, 'solver': 'adam'}
Accuracy: 0.78
"""

"""
Precision для тренировочных данных  0.8693652253909844
Accuracy для тренировочных данных  0.7898040707627925
Recall для тренировочных данных  0.2294803302574065
F1-score для тренировочных данных  0.36311239193083567

Precision для тестовых данных  0.5962962962962963
Accuracy для тестовых данных  0.7689576464620848
Recall для тестовых данных  0.1671858774662513
F1-score для тестовых данных  0.2611516626115166


------------------------------------------------

TP  997
FP  63
TN  11590
FN  3121
Precision для тренировочных данных  0.940566037735849
Accuracy для тренировочных данных  0.7981104559000698
Recall для тренировочных данных  0.24210781932977174
F1-score для тренировочных данных  0.3850907686365392
TP  174
FP  81
TN  2899
FN  789
Precision для тестовых данных  0.6823529411764706
Accuracy для тестовых данных  0.7793558204412884
Recall для тестовых данных  0.1806853582554517
F1-score для тестовых данных  0.2857142857142857


"""