import pandas as pd
from parse_pb_nalog import MainCompanyData
from parse_it_audit_contragent_info import ParseData
import time
import random


def set_company_statuses(inns, all_data, columns, start=0, batch_size=10):
    fileCommon = '../data/data_target_addit_info/data(22-23).csv'
    results = []
    results_all = []
    browser = None

    print('indexes: ', start, ' to: ', start + batch_size)

    for index in range(start, start + batch_size):
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

        except Exception:
            print(index, inn, Exception)
            currentResult = [None, None, None, None, None, None]
            currentResultAll = [None, None, None, None, None]

        results.append(currentResult)
        results_all.append(currentResultAll)
    # print(results_all)

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
                              error_bad_lines=False, names=columns)

    targeted_df = targeted_df.append(df_new_target, ignore_index=True)
    targeted_df.to_csv(fileCommon, index=False, header=False)


fileColumnsName = '../data/structure-2018-data.csv'
df = pd.read_csv(fileColumnsName, encoding='ISO-8859-1', engine='python')

columns = list(df['Columns'])

fileName = '../data/targeted/pre-companies-2018-1 (target)22.csv'
df = pd.read_csv(fileName, encoding='utf-8', engine='python', names=columns, header=None, delimiter=',')


batchSize = 50
countCycles = 20

for i in range(countCycles):
    time.sleep(random.randint(5, 10))
    set_company_statuses(df['INN'], df, columns, 0 + i*batchSize, batchSize)

