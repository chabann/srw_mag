from parse_bfo_nalog import FinancialCompanyData
from data_writer import DataWriter
from QuantitativeBankruptcyModels.twofactor_altman import TwoFactorAltman
from QuantitativeBankruptcyModels.nonmanufacturing_altman import NonManufacturingAltman
from QuantitativeBankruptcyModels.taffler import Taffler


class PerformanceIndicators:
    def __init__(self, finance_data):
        self.financeData = finance_data
        self.indicators = {}
        self.indicatorsNames = {
            'profit_1': 'Прибыль (убыток) от продаж',
            'netProfit_2': 'Чистая прибыль (убыток) ',
            'actives_3': 'Активы',
            'equityCapital_4': 'Собственный капитал',
            'profitBeforeTaxes_5': 'Прибыль до уплаты процентов и налогов',
            'netOperatingProfit_6': 'Чистая операционная прибыль уменьшенная на скорректированные налоги',
            'netOperatingProfitTax_7': 'Чистая операционная прибыль после налогообложения',
            'salesProfit_8': 'Рентабельность продаж',
            'mainActProfit_9': 'Рентабельность основной деятельности',
            'netProfit_10': 'Чистая рентабельность',
            'ecoActProfit_11': 'Экономическая рентабельность активов',
            'netProfitEquity_12': 'Чистая рентабельность собственного капитала',
        }

        self.set_indicators()
        self.check_dynamic()
        self.mergedData = self.prepare_data()

    def set_indicators(self):
        balance = self.financeData['balance']
        fin_report = self.financeData['financialResult']
        # 'beforePrevious'
        periods = ['current', 'previous']
        for period in periods:
            current = {}

            income_tax = fin_report.get(period + '2410', 0) + fin_report.get(period + '2430', 0) + fin_report.get(
                period + '2450', 0) + fin_report.get(period + '2460', 0)

            current['profit_1'] = fin_report.get(period + '2200', 0)
            current['netProfit_2'] = fin_report.get(period + '2400', 0)
            current['actives_3'] = fin_report.get(period + '2400', 0)
            current['equityCapital_4'] = balance.get(period + '1300')
            current['profitBeforeTaxes_5'] = fin_report.get(period + '2300', 0) + fin_report.get(period + '2330', 0)

            try:
                income_tax_rate = income_tax / fin_report.get(period + '2300', 1)
                current['netOperatingProfit_6'] = fin_report.get(period + '2200', 0) - income_tax + (
                        fin_report.get(period + '2330', 0) * (1 - income_tax_rate))

                current['netOperatingProfitTax_7'] = fin_report.get(period + '2200', 0) * (1 - income_tax_rate)

                current['salesProfit_8'] = fin_report.get(period + '2100', 0) / fin_report.get(period + '2110', 1) * 100
                current['mainActProfit_9'] = fin_report.get(period + '2200', 0) / fin_report.get(period + '2110', 1) * 100
                current['netProfit_10'] = fin_report.get(period + '2400', 0) / fin_report.get(period + '2110', 1) * 100
                current['ecoActProfit_11'] = fin_report.get(period + '2200', 0) / balance.get(period + '1600', 1) * 100
                current['netProfitEquity_12'] = fin_report.get(period + '2200', 0) / balance.get(period + '1600', 1) * 100
            except ZeroDivisionError:
                print('Заполнены не все данные')
            self.indicators[period] = current

    def check_dynamic(self):
        for indicator in self.indicatorsNames:
            if indicator not in self.indicators['current']:
                self.indicators['current'][indicator] = None
            if indicator not in self.indicators['previous']:
                self.indicators['previous'][indicator] = None

            if (self.indicators['current'][indicator] is not None) \
                    & (self.indicators['previous'][indicator] is not None):
                if self.indicators['current'][indicator] - self.indicators['previous'][indicator] <= 0:
                    print('Показатель', self.indicatorsNames[indicator], 'увеличивается')
                else:
                    print('Показатель', self.indicatorsNames[indicator], 'уменьшается, это является негативным фактором')

    def prepare_data(self):
        merged_data = self.financeData['balance']
        merged_data.update(self.financeData['financialResult'])
        merged_data.update(self.financeData['additData'])
        return merged_data


searchRequestLine = {'query': '', 'inn': '2801217136', 'name': '', 'ogrn': ''}
financeCompData = FinancialCompanyData(searchRequestLine['query'], searchRequestLine['inn'])
financeData = financeCompData.companyData
financeData['additData'] = financeCompData.additCompanyData

# расчет показателей
if financeData:
    performance = PerformanceIndicators(financeData)

    # запись в файл
    writer = DataWriter(performance.mergedData)
    writer.update()

    # Количественные модели банкротства
    # TwoFactorAltman(performance.financeData)
    # NonManufacturingAltman(performance.financeData)
    # Taffler(performance.financeData)
