import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from algorithmMaximumProbability import AlgorithmMaximumProbability
from analyze_separating import Separating

import six
import sys

sys.modules['sklearn.externals.six'] = six

from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt


class IndicatorAnalize:
    def __init__(self, inn, group_count, is_use_old):
        self.data = {}
        self.nnlist = []
        self.nndict = {}
        self.inn = inn
        self.inns = []
        self.m = group_count
        self.is_use_old = is_use_old

        self.colors = {
            0: {  # group
                0: 'green',
                1: 'red'
            },
            1: {
                0: 'olive',
                1: 'magenta'
            },
            2: {
                0: 'mediumseagreen',
                1: 'maroon'
            },
            3: {  # group
                0: 'green',
                1: 'red'
            },
            4: {
                0: 'olive',
                1: 'magenta'
            },
            5: {
                0: 'mediumseagreen',
                1: 'maroon'
            },
            6: {  # group
                0: 'green',
                1: 'red'
            },
            7: {
                0: 'olive',
                1: 'magenta'
            },
            8: {
                0: 'mediumseagreen',
                1: 'maroon'
            },
            9: {  # group
                0: 'green',
                1: 'red'
            },
            10: {
                0: 'olive',
                1: 'magenta'
            },
            11: {
                0: 'mediumseagreen',
                1: 'maroon'
            },
            12: {
                0: 'olive',
                1: 'magenta'
            }
        }

        if is_use_old:
            self.data_types = [2013, 2014, 2015, 2016, 2017, 2018, 'Common']
            self.years = [2013, 2014, 2015, 2016, 2017, 2018]

            self.file_name = f'data/group_find_opt/group4_all_4.csv'
        else:
            self.data_types = [2013, 2014, 2015, 2016, 2017, 2018, 'Common']
            self.years = [2013, 2014, 2015, 2016, 2017, 2018]

            self.file_name = f'data/group_worse/new_2013_group_{self.m}_#group#.csv'

        if is_use_old:
            self.columns = ['inn', 'Label', 'Old', 'Type'] + [str(year) for year in self.years]
        else:
            self.columns = ['inn', 'Label', 'Old'] + [str(year) for year in self.years]

        self.N = len(self.years)

        self.z_df = []
        self.statistic = {group: [] for group in range(self.m)}
        self.group = 0

        self.current_inn = 0
        self.bankrupt_count = 0
        self.non_bankrupt_count = 0

        self.bankrupt_count_check = 0
        self.non_bankrupt_count_check = 0

        self.read_indicators()

        self.n = len(self.data)

    def maximum_likelihood(self):
        print('Maximum Likelihood')

        if self.is_use_old:
            columns = ['Label', 'Old', 'Type'] \
                        + [f'group_{group + 1}_{str(year)}' for year in self.years for group in range(self.m)]
        else:
            columns = ['Label'] + [f'group_{group + 1}_{str(year)}' for year in self.years for group in range(self.m)]

        df = pd.DataFrame(self.nnlist, columns=columns)

        df1 = df.drop('Label', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(df1, df['Label'], test_size=0.2)
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_test = x_test.to_numpy()
        y_test = y_test.to_numpy()

        print('Probit')
        print('')

        fit_object = AlgorithmMaximumProbability(x_train, y_train)
        fit_object.process_probit('Nelder-Mead')

        datas = [(x_train, y_train), (x_test, y_test)]
        names = ['тренировочных', 'тестовых']

        for i in range(len(datas)):
            predicted_val = fit_object.predict(datas[i][0], 'probit')

            print('Precision для ' + names[i] + ' данных ', precision_score(datas[i][1], predicted_val))
            print('Accuracy для ' + names[i] + ' данных ', accuracy_score(datas[i][1], predicted_val))
            print('F1-score для ' + names[i] + ' данных ', f1_score(datas[i][1], predicted_val))
            print('')

        print('Logit')
        print('')
        fit_object = AlgorithmMaximumProbability(x_train, y_train)
        fit_object.process_logit('Nelder-Mead')

        for i in range(len(datas)):
            predicted_val = fit_object.predict(datas[i][0], 'logit')

            print('Precision для ' + names[i] + ' данных ', precision_score(datas[i][1], predicted_val))
            print('Accuracy для ' + names[i] + ' данных ', accuracy_score(datas[i][1], predicted_val))
            print('F1-score для ' + names[i] + ' данных ', f1_score(datas[i][1], predicted_val))
            print('')

    def read_indicators(self):
        for group_id in range(self.m):
            count = 0

            filename = self.file_name.replace('#group#', str(group_id))
            df = pd.read_csv(filename, encoding='utf-8', engine='python', names=self.columns,
                             delimiter=',', on_bad_lines='skip')

            df = df.reindex(np.random.permutation(df.index))

            for index, row in df.iterrows():
                count += 1
                if group_id == 0:
                    if row['inn'] not in self.data:
                        self.data[row['inn']] = {}

                    if row['Label'] == 1:
                        self.bankrupt_count += 1
                    else:
                        self.non_bankrupt_count += 1

                    self.data[row['inn']][group_id] = row

                    arr_list = []
                    self.nndict[row['inn']] = []
                    for key, value in row.items():
                        if key != 'inn':
                            arr_list.append(value)
                    self.nndict[row['inn']].append(arr_list)
                else:
                    if row['inn'] in self.data:
                        self.data[row['inn']][group_id] = row

                        arr_list = []
                        for key, value in row.items():
                            if (key != 'inn') and (key != 'Label') and (key != 'Old') and (key != 'Type'):
                                arr_list.append(value)
                        self.nndict[row['inn']].append(arr_list)

        num = 0
        for inn in self.nndict:
            self.nnlist.append([])
            for group_id in range(self.m):
                self.nnlist[num] += self.nndict[inn][group_id]
            num += 1

        print(f'Bankrupts count: {self.bankrupt_count}')
        print(f'Non Bankrupts count: {self.non_bankrupt_count}')

    def print_graf(self):
        figure_num = 0

        for group_id in range(self.m):
            plt.figure(figure_num)
            plt.grid()
            figure_num += 1

            for inn in self.data:

                row = self.data[inn][group_id]
                color = self.colors[group_id][row['Label']]

                vals = []
                for key, value in row.items():
                    if (key != 'Label') and (key != 'inn') and (key != 'Old'):
                        vals.append(value)

                plt.plot(self.years, vals, color=color, marker='.', linestyle='-',
                         linewidth=1, markersize=3)

        plt.show()


groups = 1
process = IndicatorAnalize(None, groups, True)

# columns = ['Old', 'Type', 'group_1_2016', 'group_1_2017', 'group_1_2018']
columns = False
separate = Separating(process, 250, columns)
# separate.plot_gist_age()
separate.all(False)
# separate.separate_by_type(False)

