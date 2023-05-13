import pandas as pd
from parse_it_audit_contragent_info import ParseData
import time
import random


def set_company_statuses(inns, all_data, columns, start=0, batch_size=10):
    fileCommon = '../data/data_target_addit_info/data_2012_targeted_3.csv'
    results = []
    results_all = []
    browser = None

    for index in range(start, start + batch_size):
        if index not in inns:
            continue

        inn = inns[index]
        try:
            time.sleep(random.randint(3, 5))

            audit_info = ParseData(str(inn), browser).check_company_status()
            browser = audit_info['browser']
            currentResult = [inn, audit_info['status'], audit_info['statusText'], audit_info['date'],
                             audit_info['age'], audit_info['empNumber'], audit_info['firmScale']]
            currentResultAll = [audit_info['status'], audit_info['statusText'], audit_info['date'],
                                audit_info['age'], audit_info['empNumber'], audit_info['firmScale']]
            print(index, currentResult)

        except Exception as e:
            print(index, inn, e)
            currentResult = [None, None, None, None, None, None]
            currentResultAll = [None, None, None, None, None]

        results.append(currentResult)
        results_all.append(currentResultAll)

    if 'Status' not in columns:
        columns += ['Status', 'TextStatus', 'Date', 'Age', 'EmpNumber', 'FirmScale']

    df_new_target = all_data.iloc[start:start + batch_size]
    df_add = pd.DataFrame(results_all, columns=['Status', 'TextStatus', 'Date', 'Age', 'EmpNumber', 'FirmScale'])

    df_new_target.insert(len(df_new_target.columns), 'Status', df_add['Status'].to_list())
    df_new_target.insert(len(df_new_target.columns), 'TextStatus', df_add['TextStatus'].to_list())
    df_new_target.insert(len(df_new_target.columns), 'Date', df_add['Date'].to_list())
    df_new_target.insert(len(df_new_target.columns), 'Age', df_add['Age'].to_list())
    df_new_target.insert(len(df_new_target.columns), 'EmpNumber', df_add['EmpNumber'].to_list())
    df_new_target.insert(len(df_new_target.columns), 'FirmScale', df_add['FirmScale'].to_list())

    targeted_df = pd.read_csv(fileCommon, encoding='utf-8', engine='python', header=None, delimiter=',',
                              on_bad_lines='skip', names=columns)

    targeted_df = targeted_df.append(df_new_target, ignore_index=True)
    targeted_df.to_csv(fileCommon, index=False, header=False)


fileColumnsName = '../data/structure-2018-data.csv'
df = pd.read_csv(fileColumnsName, encoding='ISO-8859-1', engine='python')

columns = list(df['Columns'])

fileName = '../data/2012/data_2012_3.csv'
df = pd.read_csv(fileName, encoding='utf-8', engine='python', names=columns, header=None,
                 delimiter=';', on_bad_lines='skip', index_col=False)


batchSize = 100
countCycles = 10
start = 9000

for i in range(countCycles):
    time.sleep(random.randint(1, 5))
    set_company_statuses(df['INN'], df, columns, start + i * batchSize, batchSize)
