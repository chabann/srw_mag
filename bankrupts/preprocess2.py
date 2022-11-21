import pandas as pd
import numpy as np


def preprocess2(columns_file, data_file):
    df = pd.read_csv(columns_file, encoding='ISO-8859-1', engine='python')
    columns = list(df['Columns'])

    df = pd.read_csv(data_file, encoding='utf-8', engine='python', names=columns, header=None, delimiter=',',
                     error_bad_lines=False)

    # Type: 0 -social-enterprise; 1 - small corporate; 2 - full
    columnsToDrop = ['None', 'Name', 'OKPO Code', 'OKOPF Code', 'OKFS Code', 'OKBED Code', 'Code value', 'INN',
                     'StatusText', 'StatusComment', 'Type']
    df.drop(columnsToDrop, inplace=True, axis=1)

    newcols = ['feature1', 'feature2', 'feature3', 'feature4',
               'feature5', 'feature6', 'feature7', 'feature8',
               'feature9', 'feature10', 'feature11', 'feature12',
               'feature12', 'feature22', 'feature32', 'feature42',
               'feature52', 'feature62', 'feature72', 'feature82',
               'feature92', 'feature102', 'feature112', 'feature122', 'Label']
    df_add = pd.DataFrame([], columns=['Status', 'TextStatus', 'Date'])

    for col in df.columns:
        if ('(' in col) and (')' in col):
            ind1 = col.find('(')
            ind2 = col.find(')')
            string = col[0: ind1 - 1]

            year = col[ind1 + 1:ind2]
            if int(year) <= 2016:
                df.drop(col, inplace=True, axis=1)

            """if int(year) == 2018:
                if (string + ' (2017)') in df.columns:"""

    df['nalog (2018)'] = df['2410 (2018)'] + df['2430 (2018)'] + df['2450 (2018)'] + df['2460 (2018)']
    df['st_nalog (2018)'] = df['nalog (2018)'] / df['2300 (2018)']

    df['nalog (2017)'] = df['2410 (2017)'] + df['2430 (2017)'] + df['2450 (2017)'] + df['2460 (2017)']
    df['st_nalog (2017)'] = df['nalog (2017)'] / df['2300 (2017)']

    df['feature1'] = df['2100 (2018)'] - df['2210 (2018)'] - df['2220 (2018)']
    df['feature2'] = df['2300 (2018)'] - df['2410 (2018)'] + df['2430 (2018)'] + df['2450 (2018)'] + df['2460 (2018)']
    df['feature3'] = df['1600 (2018)']
    df['feature4'] = df['1300 (2018)']
    df['feature5'] = df['2300 (2018)'] + df['2330 (2018)']
    df['feature6'] = df['2200 (2018)'] - df['nalog (2018)'] + (df['2330 (2018)'] * (1 - df['st_nalog (2018)']))
    df['feature7'] = df['2200 (2018)'] * (1 - df['st_nalog (2018)'])
    df['feature8'] = df['2100 (2018)'] / df['2110 (2018)']
    df['feature9'] = df['2200 (2018)'] / df['2110 (2018)']
    df['feature10'] = df['2400 (2018)'] / df['2110 (2018)']
    df['feature11'] = df['2200 (2018)'] / df['1600 (2018)']
    df['feature12'] = df['2400 (2018)'] / df['1300 (2018)']

    df['feature12'] = df['2100 (2017)'] - df['2210 (2017)'] - df['2220 (2017)']
    df['feature22'] = df['2300 (2017)'] - df['2410 (2017)'] + df['2430 (2017)'] + df['2450 (2017)'] + df['2460 (2017)']
    df['feature32'] = df['1600 (2017)']
    df['feature42'] = df['1300 (2017)']
    df['feature52'] = df['2300 (2017)'] + df['2330 (2017)']
    df['feature62'] = df['2200 (2017)'] - df['nalog (2017)'] + (df['2330 (2017)'] * (1 - df['st_nalog (2017)']))
    df['feature72'] = df['2200 (2017)'] * (1 - df['st_nalog (2017)'])
    df['feature82'] = df['2100 (2017)'] / df['2110 (2017)']
    df['feature92'] = df['2200 (2017)'] / df['2110 (2017)']
    df['feature102'] = df['2400 (2017)'] / df['2110 (2017)']
    df['feature112'] = df['2200 (2017)'] / df['1600 (2017)']
    df['feature122'] = df['2400 (2017)'] / df['1300 (2017)']

    df1 = df[newcols]

    """df1 = df.drop('Label', axis=1)
    row_sum = df1.sum(axis=1)
    rows_to_del = []
    for i in range(df.shape[0]):
        val = row_sum[i]
        if val == 0:
            rows_to_del.append(i)

    df.drop(df.index[rows_to_del], inplace=True)"""

    df1 = df1.replace([np.inf, -np.inf], np.nan)
    df1 = df1.dropna()
    return df1
