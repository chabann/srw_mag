import pandas as pd
import re

"""
Методы для преобработки напарсенных данных в готовые для использования
"""


class PreprocessParseDataToUse:
    def __init__(self, columns_file, data_file, target_file):
        self.columns_file = columns_file
        self.data_file = data_file
        self.target_file = target_file

        self.columnsToDrop = ['None', 'Name', 'OKPO Code', 'OKOPF Code', 'OKFS Code', 'People']
        self.moneyColumns = [
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
        self.df = None

    def read_file(self):
        dfColumns = pd.read_csv(self.columns_file, encoding='ISO-8859-1', engine='python')
        columns = list(dfColumns['Columns'])

        self.df = pd.read_csv(self.data_file, encoding='utf-8', engine='python', names=columns, header=None,
                              delimiter=',', error_bad_lines=False)

        # Type: 0 -social-enterprise; 1 - small corporate; 2 - full

        self.df.drop(self.columnsToDrop, inplace=True, axis=1)

    def preprocess(self):
        self.read_file()

        self.from_years_to_codes()
        # self.delete_empty()  #
        self.prepare_old()
        # self.prepare_okved()  #
        # self.set_values_size()  #
        self.set_when_bankrupt()

    def prepare_okved(self):
        for index, row in self.df.iterrows():
            okvedCode = row['OKBED Code']
            arCodes = okvedCode.split('.')
            codeValue = 0
            for i in range(len(arCodes)):
                if i == 0:
                    codeValue += int(arCodes[i]) * 10000
                elif i == 1:
                    codeValue += int(arCodes[i]) * 100
                else:
                    codeValue += int(arCodes[i])
            self.df.loc[index, 'OKBED Code'] = codeValue

    def prepare_old(self):
        for index, row in self.df.iterrows():
            year = row['Old']
            if (year is None) or (str(year) == 'nan'):
                self.df.loc[index, 'Old'] = 0
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

                self.df.loc[index, 'Old'] = old

    def set_when_bankrupt(self):
        for index, row in self.df.iterrows():
            status_text = row['StatusText']
            if status_text == 'действующая':
                self.df.loc[index, 'DateLiquid'] = ''
            elif status_text == 'ликвидирована':
                status_comment = row['StatusComment']

                date_liquid = status_comment.replace('Организация ликвидирована ', '').split('.')[-2]
                self.df.loc[index, 'DateLiquid'] = date_liquid
            else:
                self.df.drop(index, inplace=True)

        self.df.drop('StatusText', inplace=True, axis=1)
        self.df.drop('StatusComment', inplace=True, axis=1)

        self.df.dropna(inplace=True)

    def from_years_to_codes(self):
        """
        Преобразуем датасет - избавляемся от года в колонке,
        вместо этого получаем разницу 2018 и 2017 года
        """
        columns = self.df.columns

        for col in columns:
            if ('(' in col) and (')' in col):
                ind1 = col.find('(')
                ind2 = col.find(')')
                string = col[0: ind1 - 1]

                year = col[ind1 + 1: ind2]
                if int(year) <= 2016:
                    self.df.drop(col, inplace=True, axis=1)

                if int(year) == 2018:
                    if (string + ' (2017)') in columns:
                        # 2017 есть, берем разницу между ними, чтобы получить динамику изменения
                        self.df[string] = self.df[col] - self.df[string + ' (2017)']

                        self.df.drop(string + ' (2017)', inplace=True, axis=1)
                    else:
                        # 2017 нет, оставляем за один, 2018, год
                        self.df[string] = self.df[col]
                    self.df.drop(col, inplace=True, axis=1)

    def delete_empty(self):
        indexes_to_drop = []
        for index, row in self.df.iterrows():
            isEmpty = True
            isNan = True
            for col in self.moneyColumns:
                if row[col] is not None:
                    isNan = False
                if row[col] != 0:
                    isEmpty = False
            if ((row['Label'] != 0) & (row['Label'] != 1)) | isEmpty | isNan:
                indexes_to_drop.append(index)

        for index in indexes_to_drop:
            self.df.drop(index, inplace=True)

        # self.df.dropna(inplace=True)

    def set_values_size(self):
        """
        Code:
        383 - рубли
        384 - тыс. рублей
        385 - млн. рублей
        """
        for index, row in self.df.iterrows():
            if row['Code value'] == 384:  # Измеряется в тысячах
                for col in self.moneyColumns:
                    self.df.loc[index, col] = row[col] * 1000
            elif row['Code value'] == 385:  # Измеряется в миллионах
                for col in self.moneyColumns:
                    self.df.loc[index, col] = row[col] * 1000000

        self.df.drop('Code value', inplace=True, axis=1)

    def write(self, right_columns_file):
        df = pd.read_csv(right_columns_file, encoding='ISO-8859-1', engine='python')
        columns = list(df['Columns'])

        self.df = self.df[columns]

        written_df = pd.read_csv(self.target_file, encoding='utf-8', engine='python',
                                 header=None, delimiter=',',
                                 error_bad_lines=False, names=columns)

        written_df = written_df.append(self.df, ignore_index=True)
        written_df.to_csv(self.target_file, index=False, header=False)


preprocessing = PreprocessParseDataToUse('data/columns_all.csv', 'data/data_2012_targeted_3.csv', 'data/data_hovanov_new2.csv')
preprocessing.preprocess()
preprocessing.write('data/columns_to_hovanov.csv')
