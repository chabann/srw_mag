import pandas as pd


class DataWriter:
    def __init__(self, data={}):
        self.fileName = '../data/testFinanceData.csv'
        self.data = data
        self.preparedData = self.prepare()

    def prepare(self):
        prepared = {
            'columns': [],
            'values': [],
        }
        for key in self.data:
            prepared['columns'].append(key)
            prepared['values'].append(self.data[key])
        return prepared

    def update(self):
        df = pd.read_csv(self.fileName)
        df = df.append(self.data, ignore_index=True)
        df.to_csv(self.fileName, index=False)

    def create(self):
        df = pd.DataFrame([self.preparedData['values']],
                          columns=self.preparedData['columns'])
        df.to_csv(self.fileName, index=False)

    def read(self):
        df = pd.read_csv(self.fileName)
        print(df)


# dataTable = DataWriter()
# dataTable.read()
