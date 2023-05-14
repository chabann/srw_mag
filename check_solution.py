from parse_bfo_nalog import FinancialCompanyData
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from parse_it_audit_contragent_info import ParseData
import time
import random
import pandas as pd
import numpy as np
from utils import count_min_max
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
import pickle
import re


def classic_component_check(divisible, divider):
    if divider != 0:
        return divisible / divider
    else:
        return 0


def write_classic_results(temp_class, y):
    print('Precision  ', precision_score(temp_class, y))
    print('Accuracy ', accuracy_score(temp_class, y))
    print('F1-score ', f1_score(temp_class, y))
    print('Recall ', recall_score(temp_class, y))
    print('Roc-Auc-score ', roc_auc_score(temp_class, y))
    print()


def get_feature_val(finance_data, col, feature):
    feature_val = 0

    if 'balance' in finance_data:
        if feature in finance_data['balance']:
            feature_val = finance_data['balance']['current' + col]

    if 'financialResult' in finance_data:
        if feature in finance_data['financialResult']:
            feature_val = finance_data['financialResult']['current' + col]

    if 'fundsMovement' in finance_data:
        if feature in finance_data['fundsMovement']:
            feature_val = finance_data['fundsMovement']['current' + col]

    if 'targetedFundsUsing' in finance_data:
        if feature in finance_data['targetedFundsUsing']:
            feature_val = finance_data['targetedFundsUsing']['current' + col]

    if feature_val is None:
        feature_val = 0
    else:
        feature_val *= 1000

    return feature_val


def set_age(year):
    if (year is None) or (str(year) == 'nan'):
        age = 0
    else:
        years = re.finditer('[0-9]+ ', year)
        order = 0
        old = 0
        for entry in years:
            num = year[entry.start(): entry.end()]

            if order == 0:
                old = int(num)
            else:
                old += int(num) / 12
            order += 1

        age = old
    return age


def get_bankrupts():
    columns = ['inn']

    target_filename = 'data/prepared/timelyStay/bankrupts_test.csv'
    target_columns = ['INN', 'Type', 'Label', 'DateLiquid', 'Old']

    file_name = 'data/prepared/timelyStay/inns_test.csv'
    df = pd.read_csv(file_name, encoding='utf-8', engine='python', names=columns, header=None,
                     delimiter=';', on_bad_lines='skip', index_col=False)

    browser = None

    for index, row in df.iterrows():
        inn = str(int(row['inn']))

        try:
            time.sleep(random.randint(2, 3))

            audit_info = ParseData(str(inn), browser).check_company_status()
            print(audit_info)
            browser = audit_info['browser']

            label = int(float(audit_info['status']))
            date_liquid = ''
            if label == 1:
                status_comment = audit_info['date']
                date_liquid = status_comment.replace('Организация ликвидирована ', '').split('.')[-2]
                date_liquid = re.findall("\d+", date_liquid)[0]

            current_result = [[inn, audit_info['firmScale'], label, date_liquid, set_age(audit_info['age'])]]

            written_df = pd.read_csv(target_filename, encoding='utf-8', engine='python', names=target_columns,
                                     header=None, delimiter=',', on_bad_lines='skip')

            df = pd.DataFrame(current_result, columns=target_columns)
            df = df.append(written_df, ignore_index=True)

            df.to_csv(target_filename, index=False, header=False)
        except Exception as e:
            print(e)


class CheckTimeStay:
    def __init__(self):
        self.filename = 'data/classic_methods_companies.csv'
        self.target_filename = 'data/classic_methods_data.csv'
        self.columns_file = 'data/columns_to_hovanov.csv'
        self.columns_si = ['inn', 'Label', 'Old', 'Type'] + [str(i) for i in range(6)]
        self.file = 'data/data_hovanov_new2.csv'

        self.columns_no_money = [
            'Label',
            'DateLiquid',
            'Old',
            'INN',
            'Type',
            'Size',
            'Code value',
        ]
        self.columns_stay = ['INN', 'Type', 'Label', 'DateLiquid', 'Old']
        self.groups = [
            [4313, 4100, 3310, 4300, 3314, 3316, 1120, 6210, 4490, 1430, 2210, 1310, 1130, 1450, 4319, 4212, 6325,
             6350, 1370, 1230, 3320, 1250, 4321, 6230, 1150, 1260, 4312, 2310]
        ]
        # self.groups = [
        #     [
        #         1100, 1200, 1300, 1400, 1500, 2100, 2200, 2300, 2400, 2500, 6100, 6200, 6300, 6400, 3200, 3300,
        #         3600, 4100, 4200, 4300, 4400
        #     ]
        # ]

        self.features_classic = [1200, 1500, 1400, 1700, 1600, 1370, 2300, 1300, 2330, 2110, 2200]

        self.group = 0
        self.m = len(self.groups)
        self.df = None
        self.company_statistic = None
        self.years = []
        self.N = len(self.years)
        self.z_df = []
        self.summary_indicators = {group: [] for group in range(self.m)}
        self.inn = 0

    def prepare_income_data(self):
        df_columns = pd.read_csv(self.columns_file, encoding='ISO-8859-1', engine='python')
        columns = list(df_columns['Columns'])

        self.df = pd.read_csv(self.file, encoding='utf-8', engine='python', names=columns, header=None,
                              delimiter=',', on_bad_lines='skip')

        for column in self.df.columns:
            if column not in self.columns_stay:
                self.df.drop(column, inplace=True, axis=1)

        bankrupt_count = 0
        non_bankrupt_count = 0

        max_bankrupts = 0  # 120
        max_non_bankrupt = 300

        self.df = shuffle(self.df)

        for index, row in self.df.iterrows():
            try:
                if (row['Label'] == 1) and (int(float(row['DateLiquid'])) == 2023):
                    if bankrupt_count < max_bankrupts:
                        bankrupt_count += 1
                    else:
                        self.df.drop(index, inplace=True)
                elif row['Label'] == 0:
                    if non_bankrupt_count < max_non_bankrupt:
                        non_bankrupt_count += 1
                    else:
                        self.df.drop(index, inplace=True)
                else:
                    self.df.drop(index, inplace=True)
            except Exception as e:
                self.df.drop(index, inplace=True)

        self.df.to_csv(self.filename, index=False, header=False)

    def read_prepared(self):
        self.df = pd.read_csv(self.filename, encoding='utf-8', engine='python', names=self.columns_stay, header=None,
                              delimiter=',', on_bad_lines='skip')

    def get_data(self):
        for index, row in self.df.iterrows():
            time.sleep(random.randint(2, 3))
            inn = str(int(row['INN']))

            try:
                search_request_line = {'query': '', 'inn': inn, 'name': '', 'ogrn': ''}
                finance_comp_data = FinancialCompanyData(search_request_line['query'], search_request_line['inn'])
                self.inn = inn

                common_data = finance_comp_data.additCompanyData
                reports = finance_comp_data.detail_reports
                company_statistic = {
                    'Common': {
                        'Old': row['Old'],
                        'Type': row['Type'],
                        'Label': row['Label'],
                        'INN': inn
                    },
                    'datas': {}
                }
                self.years = []

                rep_years = {}
                for report_id in range(len(reports)):
                    report = reports[report_id]
                    rep_years[int(float(report['period']))] = report_id

                report_order = sorted(rep_years, reverse=True)

                for report_id in range(len(reports)):
                    report = reports[rep_years[report_order[report_id]]]

                    finance_comp_data.send_detail_request(report['id'])
                    finance_data = finance_comp_data.companyData
                    company_statistic[report['period']] = {}

                    year = int(float(report['period']))
                    self.years.append(year)

                    for col in self.groups[0]:
                        feature = 'current' + str(col)

                        if str(col) not in company_statistic['datas']:
                            company_statistic['datas'][str(col)] = {}

                        company_statistic['datas'][str(col)][year] = get_feature_val(finance_data, str(col), feature)

                    if report_id == (len(reports) - 1):
                        year = int(float(report['period'])) - 1
                        self.years.append(year)
                        for col in self.groups[0]:
                            feature = 'previous' + str(col)
                            company_statistic['datas'][str(col)][year] = get_feature_val(finance_data, str(col), feature)

                        year = int(float(report['period'])) - 2
                        self.years.append(year)
                        for col in self.groups[0]:
                            feature = 'beforePrevious' + str(col)
                            company_statistic['datas'][str(col)][year] = get_feature_val(finance_data, str(col), feature)

                self.years = sorted(self.years)
                self.N = len(self.years)

                self.company_statistic = company_statistic
                self.normalize()
                self.minimize_dz()
                self.write_indicators()
            except Exception as e:
                print(e)

    def minimize_dz(self):
        for i in range(self.m):
            ni = len(self.groups[i])
            self.group = 0

            x0 = np.array([1] + [0 for _ in range(ni - 1)])
            constraints = [1 for _ in range(ni)]

            linear_constraint = LinearConstraint(constraints, [1], [1])
            bounds = Bounds([0 for _ in range(ni)], [1.0 for _ in range(ni)])

            res = minimize(self.dz, x0, method='trust-constr', constraints=[linear_constraint], bounds=bounds,
                           tol=1e-6, options={'gtol': 1e-6, 'disp': True})

            self.count_z_t(res.x)

    def dz(self, x):
        func_value = 0
        ni = len(self.groups[self.group])

        for year in self.years:
            year_val = 0

            for j in range(ni):
                feature = str(self.groups[self.group][j])

                alpha_j = x[j]
                y_j_t = self.company_statistic['datas'][feature][year]

                year_val += alpha_j * y_j_t

            for t in self.years:
                for j in range(len(self.groups[self.group])):
                    feature = str(self.groups[self.group][j])

                    alpha_j = x[j]
                    y_j_t = self.company_statistic['datas'][feature][t]

                    year_val -= (1 / self.N) * alpha_j * y_j_t

            func_value += year_val ** 2

        return (1 / (self.N - 1)) * func_value

    def count_z_t(self, alpha):
        z = []
        ni = len(self.groups[self.group])

        for year in self.years:
            z_year = 0

            for j in range(ni):
                feature = str(self.groups[self.group][j])

                y_j_t = self.company_statistic['datas'][feature][year]

                z_year += alpha[j] * y_j_t

            z.append(z_year)

        self.z_df.append(z + [self.company_statistic['Common']['Label']])
        comp_type = self.company_statistic['Common']['Type']
        year = self.company_statistic['Common']['Old']
        inn = self.company_statistic['Common']['INN']

        if len(z) < 6:
            z = [0 for _ in range(6 - len(z))] + z

        self.summary_indicators[self.group] = [[inn, self.company_statistic['Common']['Label'], year, comp_type] + z]

        return z

    def write_indicators(self):
        for group_id in self.summary_indicators:
            written_df = pd.read_csv(self.target_filename, encoding='utf-8', engine='python', names=self.columns_si,
                                     header=None, delimiter=',', on_bad_lines='skip')

            df = pd.DataFrame(self.summary_indicators[group_id], columns=self.columns_si)
            df = df.append(written_df, ignore_index=True)

            df.to_csv(self.target_filename, index=False, header=False)

    def normalize(self):
        for feature in self.company_statistic['datas']:
            data = self.company_statistic['datas'][feature]

            min_val = count_min_max(data, 'min')
            max_val = count_min_max(data, 'max')

            for year in data:
                cur_value = data[year]

                if min_val != max_val:
                    self.company_statistic['datas'][feature][year] = (cur_value - min_val) / (max_val - min_val)
                elif max_val == 0:
                    self.company_statistic['datas'][feature][year] = 0
                else:
                    self.company_statistic['datas'][feature][year] = 0

    def check_bankrupts(self):
        columns = ['inn', 'Label', 'Old', 'Type', 'group_1_2013', 'group_1_2014', 'group_1_2015', 'group_1_2016',
                   'group_1_2017', 'group_1_2018']

        model_name = 'final_model(All).sav'
        df = pd.read_csv(self.target_filename, encoding='utf-8', engine='python', names=columns,
                         header=None, delimiter=',', on_bad_lines='skip')

        # оставить только активные:
        # df = df[df['Label'] == 0]
        # df = df[df['Label'] == 1]

        model = pickle.load(open(model_name, 'rb'))

        y_data = df['Label']

        df1 = df.drop('inn', axis=1)
        df1 = df1.drop('Label', axis=1)

        x_data = df1

        predicted_val = model.predict(x_data)

        df['Predicted'] = predicted_val
        df_wrong = df[df['Label'] != df['Predicted']]

        print('Precision', precision_score(y_data, predicted_val))
        print('Accuracy', accuracy_score(y_data, predicted_val))
        print('F1-score', f1_score(y_data, predicted_val))
        print('Recall', recall_score(y_data, predicted_val))
        try:
            print('ROC-AUC', roc_auc_score(y_data, predicted_val))
        except Exception as e:
            print(e)

    def get_data_classic(self):
        for index, row in self.df.iterrows():
            time.sleep(random.randint(2, 3))
            inn = str(int(row['INN']))

            try:
                search_request_line = {'query': '', 'inn': inn, 'name': '', 'ogrn': ''}
                finance_comp_data = FinancialCompanyData(search_request_line['query'], search_request_line['inn'])
                self.inn = inn

                reports = finance_comp_data.detail_reports
                company_data = {
                    'Label': row['Label']
                }

                rep_years = {}
                for report_id in range(len(reports)):
                    report = reports[report_id]
                    rep_years[int(float(report['period']))] = report_id

                report_order = sorted(rep_years, reverse=True)
                report = reports[rep_years[report_order[0]]]

                finance_comp_data.send_detail_request(report['id'])
                finance_data = finance_comp_data.companyData

                for col in self.features_classic:
                    feature = 'current' + str(col)
                    company_data[str(col)] = get_feature_val(finance_data, str(col), feature)

                columns_classic = ['Label'] + self.features_classic
                company_data_list = [list(company_data.values())]

                written_df = pd.read_csv(self.target_filename, encoding='utf-8', engine='python', names=columns_classic,
                                         header=None, delimiter=',', on_bad_lines='skip')

                df = pd.DataFrame(company_data_list, columns=columns_classic)
                df = df.append(written_df, ignore_index=True)

                df.to_csv(self.target_filename, index=False, header=False)

            except Exception as e:
                print(e)

    def test_taffler(self):
        print('Taffler test')

        columns = ['Label'] + self.features_classic
        df = pd.read_csv(self.target_filename, encoding='utf-8', engine='python', names=columns,
                         header=None, delimiter=',', on_bad_lines='skip')

        a = [0.53, 0.13, 0.18, 0.16]

        y_data = []
        x_data = []

        for index, row in df.iterrows():
            params = [
                classic_component_check(row[2200], row[1500]),
                classic_component_check(row[1200], row[1400] + row[1500]),
                classic_component_check(row[1500], row[1600]),
                classic_component_check(row[2110], row[1600])
            ]

            x_data.append(params)
            y_data.append(row['Label'])

        m = len(a)
        n = len(y_data)
        z = [0 for _ in range(n)]
        predicted = []

        for i in range(len(x_data)):
            z[i] = 0
            for j in range(m):
                z[i] += a[j] * x_data[i][j]

            if z[i] <= 0.2:
                predicted.append(1)
            else:
                predicted.append(0)

        write_classic_results(predicted, y_data)

    def test_springate(self):
        print('Springate test')

        columns = ['Label'] + self.features_classic
        df = pd.read_csv(self.target_filename, encoding='utf-8', engine='python', names=columns,
                         header=None, delimiter=',', on_bad_lines='skip')

        a = [1.03, 3.07, 0.66, 0.4]

        y_data = []
        x_data = []

        for index, row in df.iterrows():
            params = [
                classic_component_check(row[1200] - row[1500], row[1600]),
                classic_component_check(row[2300] + row[2330], row[1600]),
                classic_component_check(row[2300], row[1500]),
                classic_component_check(row[2110], row[1600])
            ]

            x_data.append(params)
            y_data.append(row['Label'])

        m = len(a)
        n = len(y_data)
        z = [0 for _ in range(n)]
        predicted = []

        for i in range(len(x_data)):
            z[i] = 0
            for j in range(m):
                z[i] += a[j] * x_data[i][j]

            if z[i] < 0.862:
                predicted.append(1)
            else:
                predicted.append(0)

        write_classic_results(predicted, y_data)

    def test_2_altman(self):
        print('2 factor Altman test')

        columns = ['Label'] + self.features_classic
        df = pd.read_csv(self.target_filename, encoding='utf-8', engine='python', names=columns,
                         header=None, delimiter=',', on_bad_lines='skip')

        a = [-0.3877, -1.0736, 0.579]

        y_data = []
        x_data = []

        for index, row in df.iterrows():
            params = [
                1,
                classic_component_check(row[1200], row[1500]),
                classic_component_check(row[1400] + row[1500], row[1700])
            ]

            x_data.append(params)
            y_data.append(row['Label'])

        m = len(a)
        n = len(y_data)
        z = [0 for _ in range(n)]
        predicted = []

        for i in range(len(x_data)):
            z[i] = 0
            for j in range(m):
                z[i] += a[j] * x_data[i][j]

            if z[i] > 0:
                predicted.append(1)
            else:
                predicted.append(0)

        write_classic_results(predicted, y_data)

    def test_4_altman(self):
        print('4 factor Altman test')

        columns = ['Label'] + self.features_classic
        df = pd.read_csv(self.target_filename, encoding='utf-8', engine='python', names=columns,
                         header=None, delimiter=',', on_bad_lines='skip')

        a = [6.56, 3.26, 6.72, 1.05]

        y_data = []
        x_data = []

        for index, row in df.iterrows():
            params = [
                classic_component_check(row[1200] - row[1500], row[1600]),
                classic_component_check(row[1370], row[1600]),
                classic_component_check(row[2300], row[1600]),
                classic_component_check(row[1300], row[1400] + row[1500])
            ]

            x_data.append(params)
            y_data.append(row['Label'])

        m = len(a)
        n = len(y_data)
        z = [0 for _ in range(n)]
        predicted = []

        for i in range(len(x_data)):
            z[i] = 0
            for j in range(m):
                z[i] += a[j] * x_data[i][j]

            if z[i] <= 1.1:
                predicted.append(1)
            else:
                predicted.append(0)

        write_classic_results(predicted, y_data)


# Проверить модель на сводных показателях
# ______________________________________
# check = CheckTimeStay()
# check.check_bankrupts()
# ______________________________________

# Собрать данные для классических методов
# ______________________________________
# check = CheckTimeStay()
# check.read_prepared()
# check.get_data_classic()
# ______________________________________

# Собрать данные для классических методов
# ______________________________________
check = CheckTimeStay()
check.test_taffler()
check.test_springate()
check.test_2_altman()
check.test_4_altman()

# ______________________________________
# Собрать сводные показатели по данным
# check = CheckTimeStay()
# check.read_prepared()
# check.get_data()
# ______________________________________

# Собрать данные
# check = CheckTimeStay()
# check.prepare_income_data()
# ______________________________________

# Собрать данные из инн
# get_bankrupts()
# ______________________________________
