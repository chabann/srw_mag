from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing_data import preprocess
from preprocess2 import preprocess2
from sklearn.metrics import roc_auc_score
import pickle
from scoring import Scoring

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    # print(dataset)
    return dataset


# df = preprocess('../data/structure-2018-data.csv', '../data/targeted/pre-companies-2018-1 (target)0.csv')
# df = preprocess('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')

df_0 = preprocess2('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data.csv')
df_1 = preprocess2('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(2-3).csv')
df_2 = preprocess2('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(4-5).csv')
df_3 = preprocess2('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(6-7).csv')
df_4 = preprocess2('../data/structure-2018-data-labeled.csv', '../data/targeted/targeted-data(8-9).csv')

df = df_0.append(df_1, ignore_index=True).append(df_2, ignore_index=True).append(df_3, ignore_index=True)\
        .append(df_4, ignore_index=True)

# df = correlation(df, 0.7)

df1 = df.drop('Label', axis=1)

x_train, x_test, y_train, y_test = train_test_split(df1, df['Label'], test_size=0.2, random_state=42)

print('В выборке компаний-банкротов: ', df[df.Label == 1].shape[0])
print('В выборке не банкротов: ', df[df.Label == 0].shape[0])
print('')

classifier = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=2, kernel='rbf', degree=3, gamma='auto', probability=True,
                                                                 class_weight='balanced'))])

# regression = SVR(kernel='linear', gamma='auto', tol=0.001, C=1.0, epsilon=0.1)
# class_weight={0:1, 1:2.5}
# ('scaler', StandardScaler()),
model = classifier.fit(x_train, y_train)

datas = [(x_train, y_train), (x_test, y_test)]
names = ['тренировочных', 'тестовых']

for i in range(len(datas)):
    predicted_val = classifier.predict(datas[i][0])
    # predicted_proba = classifier.predict_proba(datas[i][0])

    # scoring = Scoring(predicted_proba, datas[i][1])
    # print('RMSE для ' + names[i] + ' данных ', scoring.rmse())

    scoring = Scoring(predicted_val, datas[i][1], True)
    print('Precision для ' + names[i] + ' данных ', scoring.precision())
    print('Accuracy для ' + names[i] + ' данных ', scoring.accuracy())
    print('Recall для ' + names[i] + ' данных ', scoring.recall())
    print('F1-score для ' + names[i] + ' данных ', scoring.f1_score())
    print('AUC-score для ' + names[i] + ' данных ', roc_auc_score(predicted_val, datas[i][1]))
    print('')


# сохраняем модель
filename = '../data/models/short_svm.sav'
pickle.dump(model, open(filename, 'wb'))
