import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import normalize
from bankrupts.max_prob.errors import Errors
from bankrupts.max_prob.algorithmMaximumProbability import AlgorithmMaximumProbability
from bankrupts.scoring import Scoring
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.svm import SVC
import xlwt
import itertools


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    corrMatrix = dataset.corr()
    for i in range(len(corrMatrix.columns)):
        for j in range(i):
            if abs(corrMatrix.iloc[i, j]) >= threshold:
                colname = corrMatrix.columns[i]  # getting the name of column
                col_corr.add(colname)

                if colname in dataset.columns:
                    del dataset[colname]  # deleting the column from the dataset

    return dataset


def prepare_income_data(dataframe):
    curmoneyColumns = []
    for col in dataframe.columns:
        if col in moneyColumns:
            curmoneyColumns.append(col)

    mdataframe = dataframe.reindex(curmoneyColumns, axis=1)
    n_size = mdataframe.shape[0]
    e_array = np.ones((n_size, 1))

    mdataframe = pd.DataFrame(data=normalize(mdataframe), columns=curmoneyColumns)

    # OKVED код 6-значный, 999999 - макс. значение
    for index, row in dataframe.iterrows():
        dataframe.loc[index, 'OKBED Code'] = row['OKBED Code'] / 999999

    for col in mdataframe.columns:
        dataframe[col] = mdataframe[col]

    dataframe['coeff'] = e_array

    # TODO: удалять те слобцы, где сумма модулей близка к 0, или сумма строк != 0 очень мала
    # TODO: просчитать парные коэффициенты корреляции, оценить значимость параметров

    return dataframe


def sum_columns(dataframe, tol):
    for colname in dataframe.columns:
        value = sum([np.abs(xi_col) for xi_col in dataframe[colname]])
        if value < tol:
            del dataframe[colname]
    return dataframe


def fill_columns(dataframe, percent):
    for column in dataframe.columns:
        count = 0
        for val in dataframe[column]:
            if val != 0:
                count += 1

        if (count / dataframe.shape[0]) < percent:
            del dataframe[column]

    return dataframe


book = xlwt.Workbook()
Matrix = book.add_sheet('matrix')

row_iter = Matrix.row(0)
arNames = ['excluded Features', 'Precision train', 'Accuracy train', 'Recall train',
           'F1-score train', 'AUC-score train', 'R^2 train', 'TP / FP / TN / FN train',
           'Precision test', 'Accuracy test', 'Recall test',
           'F1-score test', 'AUC-score test', 'R^2 test', 'TP / FP / TN / FN test']

for row_ind in range(len(arNames)):
    row_iter.write(row_ind, arNames[row_ind])

moneyColumns = [
    '1110', '1120', '1130', '1140', '1150', '1160', '1170', '1180', '1190', '1100', '1210',
    '1220', '1230', '1240', '1250', '1260', '1200', '1600', '1310', '1320', '1340', '1350',
    '1360', '1370', '1300', '1410', '1420', '1430', '1450', '1400', '1510', '1520', '1530',
    '1540', '1550', '1500', '1700', '2110', '2120', '2100', '2210', '2220', '2200', '2310',
    '2320', '2330', '2340', '2350', '2300', '2410', '2421', '2430', '2450', '2460', '2400',
    '2510', '2520', '2500', '3200', '3310', '3314', '3315', '3316', '3320', '3323', '3324',
    '3325', '3326', '3330', '3300', '3600', '4110', '4111', '4112', '4113', '4119', '6400',
    '4120', '4121', '4122', '4123', '4124', '4129', '4100', '4210', '4211', '4212', '4213',
    '4214', '4219', '4220', '4221', '4222', '4223', '4224', '4229', '4200', '4310', '4311',
    '4312', '4313', '4314', '4319', '4320', '4321', '4322', '4323', '4329', '4300', '4400',
    '4490', '6100', '6210', '6215', '6220', '6230', '6240', '6250', '6200', '6310', '6311',
    '6312', '6313', '6320', '6321', '6322', '6323', '6324', '6325', '6326', '6330', '6350',
    '6300',
]

columns = ['OKBED Code', 'Type', '1150', '1210', '1220', '1230', '1240', '1260', '1200',
           '1600', '1370', '2110', '2220', '2320', '2330', '2410', '2460', '3200', '4110',
           '4119', '4124', '4100', 'Old']

rowIndex = 1
excludedCount = 1
excludedDefault = ['Size', 'Old']

sizeValues = [0, 1, 2, 3, 4, 5]
oldValues = [5, 10, 20]

df = pd.read_csv('data/columns.csv', encoding='ISO-8859-1', engine='python')
columns = list(df['Columns'])

df = pd.read_csv('data/data_prepare.csv', encoding='utf-8', engine='python', names=columns, header=None,
                 delimiter=',', error_bad_lines=False)

dfY = df[['Label']]
# del df['Label']

"""
Выбираем параметры:
"""

df = fill_columns(df, 0.05)
df = correlation(df, 0.7)
# df = sum_columns(df, 0.05)

df = prepare_income_data(df)
# print(f'Parameters: {df.columns}')
corr_matrix = df.corr()

# count_bankrupts = np.count_nonzero(y_train == 1)

# print('В тренировочной выборке компаний-банкротов: ', count_bankrupts)
# print('В тренировочной выборке не банкротов: ', len(y_train) - count_bankrupts)
# print('')

arClassifiers = []

for size in sizeValues:
    dfCurSize = df[df['Size'] == size]

    dfYCurSize = dfCurSize[['Label']]
    del dfCurSize['Label']
    del dfCurSize['Size']

    x_train, x_test, y_train, y_test = train_test_split(dfCurSize, dfYCurSize, test_size=0.2, random_state=32)

    rus = RandomUnderSampler(random_state=0)
    x_train, y_train = rus.fit_resample(x_train, y_train)

    x_train, y_train = shuffle(x_train, y_train, random_state=24)

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    maxProb = AlgorithmMaximumProbability(x_train, y_train)

    resultProbit = maxProb.process_probit('Nelder-Mead')
    # LnProbit = - resultProbit.fun
    maxProb.get_result('probit')

    resultLogit = maxProb.process_logit('Nelder-Mead')
    # LnLogit = - resultLogit.fun
    maxProb.get_result('logit')
    # maxProb.get_test_result(x_test, y_test, 'logit')

    arClassifiers.append({
        'XTrain': x_train,
        'XTest': x_test,
        'YTrain': y_train,
        'YTest': y_test,
        'Theta': resultProbit.x,
        'maxProbObj': maxProb
    })

datas = [('XTrain', 'YTrain'), ('XTest', 'YTest')]
names = ['Train', 'Test']

arValues = []
arScores = {
    'Train': {
        'Precision': [],
        'Accuracy': [],
        'Recall': [],
        'F1-score': [],
        'AUC-score': [],
        'R^2 score': [],
        'weight': [],
        'allWeight': 0,
    },
    'Test': {
        'Precision': [],
        'Accuracy': [],
        'Recall': [],
        'F1-score': [],
        'AUC-score': [],
        'R^2 score': [],
        'weight': [],
        'allWeight': 0,
    }
}
for size in sizeValues:
    for i in range(len(datas)):
        theta = arClassifiers[size]['Theta']

        xVal = arClassifiers[size][datas[i][0]]
        yVal = arClassifiers[size][datas[i][1]]

        predicted_val = arClassifiers[size]['maxProbObj'].predict(xVal, 'probit')

        predicted_proba = arClassifiers[size]['maxProbObj'].predict_proba(xVal, 'probit')

        scoring = Scoring(predicted_val, yVal, True)
        precision = scoring.precision()
        accuracy = scoring.accuracy()
        recall = scoring.recall()
        f1 = scoring.f1_score()
        if len(list(set(predicted_val))):
            roc_auc = roc_auc_score(yVal, predicted_val)
        else:
            roc_auc = 0
        r2 = r2_score(yVal, predicted_proba)
        tf = scoring.TF

        arScores[names[i]]['Precision'].append(precision)
        arScores[names[i]]['Accuracy'].append(accuracy)
        arScores[names[i]]['Recall'].append(recall)
        arScores[names[i]]['F1-score'].append(f1)
        arScores[names[i]]['AUC-score'].append(roc_auc)
        arScores[names[i]]['R^2 score'].append(r2)
        arScores[names[i]]['weight'].append(len(xVal))
        arScores[names[i]]['allWeight'] += len(xVal)


print(arScores)

for experimentType in arScores:
    scores = arScores[experimentType]

    for scoreName in scores:
        if (scoreName == 'weight') | (scoreName == 'allWeight'):
            continue

        arValues = scores[scoreName]

        mean = 0
        for size in range(len(arValues)):
            if arValues[size] != 'Undefined':
                mean += arValues[size] * scores['weight'][size]
            else:
                mean += 0
        mean /= scores['allWeight']

        print(f'{experimentType} {scoreName} = {mean}')
