import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.svm import SVC
import pickle
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import six
import sys

sys.modules['sklearn.externals.six'] = six


from sklearn.externals.six import StringIO
import pydot

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 26,
'font.family': 'Times New Roman',
'mathtext.fontset': 'cm',
'legend.loc': 'upper left'
 })
plt.rcParams["figure.figsize"] = (16, 9)


class Separating:
    def __init__(self, analyze, iters=1, use_columns=None):
        self.analyze = analyze
        self.iters = iters
        self.use_columns = use_columns

        columns = ['Label', 'Old', 'Type'] \
                  + [f'group_{group + 1}_{str(year)}' for year in self.analyze.years for group in range(self.analyze.m)]

        self.df = pd.DataFrame(self.analyze.nnlist, columns=columns)
        self.df.loc[self.df["Old"] < 40, "Old"] += 1
        self.df1 = self.df.drop('Label', axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.shuffle_data()
        self.print_count()

    def shuffle_data(self):
        if self.use_columns:
            for column in self.df1.columns:
                if column not in self.use_columns:
                    self.df1 = self.df1.drop(column, axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df1, self.df['Label'], test_size=0.2)

    def separate_by_type(self, is_greed_search):
        print('Small enterprises')
        df_small = self.df[self.df['Type'] == 1].drop('Type', axis=1)
        df1 = df_small.drop('Label', axis=1)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df1, df_small['Label'], test_size=0.2)
        self.all(is_greed_search, False)

        print('Big enterprises')
        df_big = self.df[self.df['Type'] == 2].drop('Type', axis=1)
        df1 = df_big.drop('Label', axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df1, df_big['Label'], test_size=0.2)
        self.all(is_greed_search, False)

    def all(self, is_greed_search, is_standard=True):
        self.mlp(is_greed_search)
        self.svc(is_greed_search)
        self.decision_tree(is_greed_search, is_standard)

    def print_count(self):
        print('Train: bankrupts: ', self.y_train[self.y_train == 1].shape[0])
        print('Train: Not bankrupts: ', self.y_train[self.y_train == 0].shape[0])
        print('Test: bankrupts: ', self.y_test[self.y_test == 1].shape[0])
        print('Test: Not bankrupts: ', self.y_test[self.y_test == 0].shape[0])

    def mlp(self, is_greed_search):
        print('MLP')

        if is_greed_search:
            parameters = {
                'hidden_layer_sizes': [(i, j) for i in range(20, 90, 10) for j in range(10, 60, 10)],
                'solver': ['adam'],
                'activation': ['tanh', 'logistic'],
                'max_iter': [300, 500, 1000],
                'activation': ['tanh', 'logistic'], 'solver': ['sgd', 'adam'],
            }

            mlp = MLPClassifier(max_iter=1000, alpha=0.0001, learning_rate='adaptive')

            gs = GridSearchCV(mlp, parameters)
            gs.fit(self.x_train, self.y_train)
            print(gs.best_params_)
            print(gs.best_score_)
        else:
            mlp = MLPClassifier(max_iter=1000, activation='tanh', alpha=0.0001, learning_rate='adaptive',
                                solver='adam', hidden_layer_sizes=(70, 50))

            pipe = Pipeline([('mlpc', mlp)])

            max_accuracy = 0
            max_precision = 0
            best_pipe = None

            for num in range(self.iters):
                pipe.fit(self.x_train, self.y_train)

                predicted = pipe.predict(self.x_test)
                accuracy = accuracy_score(self.y_test, predicted)
                precision = precision_score(self.y_test, predicted)

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_precision = precision
                    best_pipe = pipe

                self.shuffle_data()

            datas = [(self.x_train, self.y_train), (self.x_test, self.y_test)]
            names = ['тренировочных', 'тестовых']

            predictions = best_pipe.predict_proba(datas[1][0])
            proba_predictions = [proba[1] for proba in predictions]
            eer = self.frr_far(proba_predictions, self.y_test)

            for i in range(len(datas)):
                # predicted_val = best_pipe.predict(datas[i][0])
                predictions = best_pipe.predict_proba(datas[i][0])
                predicted_val = [1 if proba[1] > eer else 0 for proba in predictions]

                print('Precision для ' + names[i] + ' данных ', precision_score(datas[i][1], predicted_val))
                print('Accuracy для ' + names[i] + ' данных ', accuracy_score(datas[i][1], predicted_val))
                print('F1-score для ' + names[i] + ' данных ', f1_score(datas[i][1], predicted_val))
                print('Recall для ' + names[i] + ' данных ', recall_score(datas[i][1], predicted_val))
                print('ROC-AUC для ' + names[i] + ' данных ', roc_auc_score(datas[i][1], predicted_val))
                print('')

    def decision_tree(self, is_greed_search, standard=True, use_years='All'):
        print('DecisionTree')

        if is_greed_search:
            tree_params = {
                'max_depth': list(range(1, 11)) + [None],
                'class_weight': ['balanced', {0: 1, 1: 2.5}, {0: 1, 1: 2}, {0: 1, 1: 3}, None],
                'criterion': ['gini', 'entropy', 'log_loss']
            }

            tree = GridSearchCV(
                DecisionTreeClassifier(random_state=32), tree_params, cv=5
            )
        else:
            tree = DecisionTreeClassifier(max_depth=8, class_weight=None, criterion='entropy')

        if standard:
            features = ['Age', 'Type'] + [f'SI in {str(year)} year' for year in self.analyze.years for group in range(self.analyze.m)]
        else:
            features = ['Age'] + [f'SI in {str(year)} year' for year in self.analyze.years for group in range(self.analyze.m)]

        if is_greed_search:
            tree.fit(self.x_train, self.y_train)
            print("Best params:", tree.best_params_)
            print("Best cross validaton score", tree.best_score_)
        else:
            max_accuracy = 0
            max_precision = 0
            best_tree = None

            for num in range(self.iters):
                tree.fit(self.x_train, self.y_train)

                predicted = tree.predict(self.x_test)
                accuracy = accuracy_score(self.y_test, predicted)
                precision = precision_score(self.y_test, predicted)

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_precision = precision
                    best_tree = tree

                self.shuffle_data()

            datas = [(self.x_train, self.y_train), (self.x_test, self.y_test)]
            names = ['тренировочных', 'тестовых']

            predictions = best_tree.predict_proba(datas[1][0])
            proba_predictions = [proba[1] for proba in predictions]
            eer = self.frr_far(proba_predictions, self.y_test)

            for i in range(len(datas)):
                predicted_val = best_tree.predict(datas[i][0])
                # predictions = best_tree.predict_proba(datas[i][0])
                # predicted_val = [1 if proba[1] > eer else 0 for proba in predictions]

                print('Precision для ' + names[i] + ' данных ', precision_score(datas[i][1], predicted_val))
                print('Accuracy для ' + names[i] + ' данных ', accuracy_score(datas[i][1], predicted_val))
                print('F1-score для ' + names[i] + ' данных ', f1_score(datas[i][1], predicted_val))
                print('Recall для ' + names[i] + ' данных ', recall_score(datas[i][1], predicted_val))
                print('ROC-AUC для ' + names[i] + ' данных ', roc_auc_score(datas[i][1], predicted_val))
                print('')

            tag = f'new_4_test-2_tree_accuracy_{round(max_accuracy, 3)}_precision_{round(max_precision, 3)}_years_{use_years}'

            dot_data = StringIO()
            export_graphviz(best_tree, out_file=dot_data, feature_names=features,
                            class_names=['non-bankrupt', 'bankrupt'], filled=True, rounded=True,
                            special_characters=True)

            graph = pydot.graph_from_dot_data(dot_data.getvalue())
            graph[0].write_pdf(f'{tag}.pdf')

            # Save the trained model as a pickle string.
            filename = f'models/{tag}.sav'
            pickle.dump(best_tree, open(filename, 'wb'))

    def svc(self, is_greed_search):
        print('SVC')

        if is_greed_search:
            parameters = {
                'kernel': ['rbf'],  # 'sigmoid', 'linear'],
                'degree': [0, 1, 2],
                'gamma': ['auto'],  # ['scale', 'auto'],
                'C': [1, 1.5, 2, 2.5],
                'class_weight': ['balanced', {0: 1, 1: 2.5}, None]
            }

            svc = SVC()

            clf = GridSearchCV(svc, parameters)
            clf.fit(self.x_train, self.y_train)
            print(clf.best_params_)
            print(clf.best_score_)
        else:
            max_accuracy = 0
            max_precision = 0
            best_clf = None

            for num in range(self.iters):
                clf = SVC(C=2.5, class_weight=None, degree=0, gamma='auto', kernel='rbf', probability=True)
                clf.fit(self.x_train, self.y_train)

                predicted = clf.predict(self.x_test)
                accuracy = accuracy_score(self.y_test, predicted)
                precision = precision_score(self.y_test, predicted)

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_precision = precision
                    best_clf = clf

                self.shuffle_data()

            datas = [(self.x_train, self.y_train), (self.x_test, self.y_test)]
            names = ['тренировочных', 'тестовых']

            predictions = best_clf.predict_proba(datas[1][0])
            proba_predictions = [proba[1] for proba in predictions]
            eer = self.frr_far(proba_predictions, self.y_test)

            for i in range(len(datas)):
                predicted_val = best_clf.predict(datas[i][0])
                #predictions = best_clf.predict_proba(datas[i][0])
                #predicted_val = [1 if proba[1] > eer else 0 for proba in predictions]

                print('Precision для ' + names[i] + ' данных ', precision_score(datas[i][1], predicted_val))
                print('Accuracy для ' + names[i] + ' данных ', accuracy_score(datas[i][1], predicted_val))
                print('F1-score для ' + names[i] + ' данных ', f1_score(datas[i][1], predicted_val))
                print('Recall для ' + names[i] + ' данных ', recall_score(datas[i][1], predicted_val))
                print('ROC-AUC для ' + names[i] + ' данных ', roc_auc_score(datas[i][1], predicted_val))
                print('')

    @staticmethod
    def frr_far(test_predictions, labels):
        def plot_eer(eer, eer_i, fars, frrs, dists):
            ax = plt.axes()
            plt.plot(dists, frrs, 'g', label='FRR')
            plt.plot(dists, fars, 'r', label='FAR')
            plt.plot(dists[eer_i], eer, 'bo', label='EER')
            plt.legend()
            ax.set_xlabel('F(t)')
            ax.set_ylabel('Rates')

        f_target = [pred for (pred, label) in zip(test_predictions, labels) if label == 1]
        f_imposter = [pred for (pred, label) in zip(test_predictions, labels) if label == 0]

        frr_list = []
        far_list = []
        N = 100
        d = 1 / N
        border_value = np.arange(0, (d * (N + 1)), d)

        for i in range(N + 1):
            is_target = 0
            is_imposter = 0
            for pred in f_target:
                if pred < border_value[i]:
                    is_target += 1
            for pred in f_imposter:
                if pred > border_value[i]:
                    is_imposter += 1

            frr_list.append(is_target / len(f_target))
            far_list.append(is_imposter / len(f_imposter))

        # calc EER
        d_opt_index = 0
        abs_distance = len(labels)
        eer_point = 0
        for i in range(N + 1):
            a = frr_list[i]
            b = far_list[i]
            distance = abs(b - a)
            if distance < abs_distance:
                abs_distance = distance
                EER = (frr_list[i] + far_list[i]) / 2
                print('EER = ', EER)
                d_opt_index = i

                plot_eer(EER, d_opt_index, far_list, frr_list, border_value)

        return EER

    def plot_gist_age(self):
        df = self.df[self.df['Old'] < 40]

        df.loc[(df['Label'] == 1.0), 'Label'] = 'Банкроты'
        df.loc[(df['Label'] == 0.0), 'Label'] = 'Активные компании'

        sns.displot(x='Old', data=df, hue="Label", multiple="stack",
                    palette={'Банкроты': '#A52A2A', 'Активные компании': '#006400'})

        # plt.title('Распределение количества банкротов и активных компаний')
        plt.xlabel('Возраст компании')
        plt.ylabel('Количество')

        plt.tight_layout()
        plt.legend()
        # plt.savefig("D:\study\scienceWork\FIgURES\image.eps", format="eps")
        plt.show()

    def plot_gist_type(self):
        df = self.df[self.df['Type'] > 0]

        dfb = df[df['Label'] == 1]
        dfnb = df[df['Label'] == 0]

        cat_par = ['Малое предприятие', 'Крупное предприятие']
        bankr = [dfb[dfb['Type'] == 1].shape[0], dfb[dfb['Type'] == 2].shape[0]]
        nonb = [dfnb[dfnb['Type'] == 1].shape[0], dfnb[dfnb['Type'] == 2].shape[0]]

        width = 0.3

        x = np.arange(len(cat_par))
        fig, ax = plt.subplots()
        ax.bar(x - width / 2, bankr, width, label='Банкроты', color='#A52A2A')
        ax.bar(x + width / 2, nonb, width, label='Активные компании', color='#006400')

        ax.set_xticks(x)
        ax.set_xticklabels(cat_par)
        ax.legend()

        plt.xlabel('Тип компании')
        plt.ylabel('Количество')

        plt.tight_layout()
        plt.legend()
        plt.show()
