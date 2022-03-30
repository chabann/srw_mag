from selenium import webdriver


class ParseData:
    def __init__(self, link, browser):
        self.link = link
        self.browser = browser

        self.url = 'https://www.audit-it.ru/' + link
        self.browser.get(self.url)

        self.companyInfo = {}
        self.countYears = 3

        self.form1Block = self.browser.find_element_by_id('form1Block')
        self.form2Block = self.browser.find_element_by_id('form2Block')

        self.get_common_info()
        self.get_balance_data()
        self.get_finance_data()
        self.get_movement()
        self.get_indicators()

    def get_common_info(self):
        statsBlock = self.browser.find_element_by_class_name('firmStatsDescr')

        self.companyInfo['name'] = statsBlock.find_element_by_xpath("//div[@class='firmInfo'][1]/strong").text
        self.companyInfo['inn'] = statsBlock.find_element_by_xpath("//div[@class='firmInfo'][2]/strong").text
        self.companyInfo['financeData'] = {}

    def get_balance_data(self):
        balanceForm = self.browser.find_element_by_id('form1')
        self.get_table_data(balanceForm, 'class', 'calcRow', 'int')

    def get_finance_data(self):
        financeForm = self.browser.find_element_by_id('form2')
        dates = financeForm.find_elements_by_class_name('periodCell')

        self.companyInfo['years'] = ''
        for year in range(self.countYears):
            self.companyInfo['years'] += dates[year].text + ' '

        self.get_table_data(financeForm, 'class', 'calcRow', 'int')

    def get_movement(self):
        movement = self.browser.find_element_by_id('form4')
        self.get_table_data(movement, 'class', 'calcRow', 'int')

    def get_indicators(self):
        fin_indicator = self.form1Block.find_element_by_class_name('tblFin')
        tbody = fin_indicator.find_element_by_tag_name('tbody')
        self.get_table_data(tbody, 'tag', 'tr', 'float')

        fin_indicator = self.form2Block.find_element_by_class_name('tblFin')
        tbody = fin_indicator.find_element_by_tag_name('tbody')
        self.get_table_data(tbody, 'tag', 'tr', 'float')

    def get_table_data(self, block, type_select, selector, type_data):
        if type_select == 'class':
            calcRows = block.find_elements_by_class_name(selector)
        elif type_select == 'tag':
            calcRows = block.find_elements_by_tag_name(selector)
        else:
            calcRows = block.find_elements_by_css_selector(selector)

        for row in calcRows:
            tds = row.find_elements_by_css_selector('td')
            for i in range(self.countYears):
                currentVal = tds[i + 2].text.replace(' ', '').replace('*', '')
                currentVal = self.prepare_value(currentVal, type_data)

                self.companyInfo['financeData'][tds[1].text + ' (' + str(i + 1) + ')'] = currentVal

    @staticmethod
    def prepare_value(value, type_val):
        try:
            if value == '-':
                value = 0
            elif value[-1] == '%':
                value = float(value[0:-1])
            elif value[0] == '(':
                if type_val == 'int':
                    value = -int(value[1:-1])
                else:
                    value = -float(value[1:-1])
            else:
                if type_val == 'int':
                    value = int(value)
                else:
                    value = float(value)
            return value
        except IndexError:
            print('value:', value, 'type:', type_val)


path_to_chromedriver = r'D:/Downloads/chromedriver_win32/chromedriver.exe'
browserDriver = webdriver.Chrome(executable_path=path_to_chromedriver)
linkDetail = 'buh_otchet/7709383532_ooo-ernst-end-yang'

data = ParseData(linkDetail, browserDriver)
print(data.companyInfo)
