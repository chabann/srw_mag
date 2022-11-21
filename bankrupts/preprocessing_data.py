import pandas as pd


def preprocess(columns_file, data_file):
    df = pd.read_csv(columns_file, encoding='ISO-8859-1', engine='python')
    columns = list(df['Columns'])

    df = pd.read_csv(data_file, encoding='utf-8', engine='python', names=columns, header=None, delimiter=',',
                     error_bad_lines=False)

    # Type: 0 -social-enterprise; 1 - small corporate; 2 - full
    columnsToDrop = ['None', 'Name', 'OKPO Code', 'OKOPF Code', 'OKFS Code', 'OKBED Code', 'Code value', 'INN',
                     'StatusText', 'StatusComment', 'Type']
    df.drop(columnsToDrop, inplace=True, axis=1)

    for col in df.columns:
        if ('(' in col) and (')' in col):
            ind1 = col.find('(')
            ind2 = col.find(')')
            string = col[0: ind1 - 1]

            year = col[ind1 + 1:ind2]
            if int(year) <= 2016:
                df.drop(col, inplace=True, axis=1)

            """if int(year) == 2018:
                if (string + ' (2017)') in df.columns:
                    df[string] = df[col] - df[string + ' (2017)']

                    df.drop(string + ' (2017)', inplace=True, axis=1)
                df.drop(col, inplace=True, axis=1)"""

    df1 = df.drop('Label', axis=1)
    row_sum = df1.sum(axis=1)
    rows_to_del = []
    for i in range(df.shape[0]):
        val = row_sum[i]
        if val == 0:
            rows_to_del.append(i)

    df.drop(df.index[rows_to_del], inplace=True)
    df.dropna(inplace=True)
    return df
