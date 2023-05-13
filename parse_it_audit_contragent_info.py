from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import random


class ParseData:
    def __init__(self, inn, browser=None, fields=None):
        if fields is None:
            fields = []
        self.inn = inn
        self.path_to_chromedriver = r'D:/Downloads/chromedriver_win32/chromedriver.exe'

        WINDOW_SIZE = '1920,1080'
        CHROME_PATH = r'C:/Program Files/Google/Chrome/Application/chrome.exe'
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        # chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
        chrome_options.binary_location = CHROME_PATH

        if browser is not None:
            self.browser = browser
        else:
            self.browser = webdriver.Chrome(executable_path=self.path_to_chromedriver, options=chrome_options)
            # self.browser = webdriver.Chrome(options=chrome_options)

        self.linkMain = 'https://www.audit-it.ru/buh_otchet/'

        if len(fields) == 0:
            fields = self.get_fields()

        self.linkDetailBO = self.get_detail_link(fields)
        self.linkDetailCheck = self.get_detail_bo()

    def get_fields(self):
        self.browser.get(self.linkMain)

        field_names = []
        field_blocks = self.browser.find_elements_by_class_name('buhotchet-search')
        self.browser.implicitly_wait(random.randint(3, 6))

        for field in field_blocks:
            field_names.append(field.get_attribute('name'))
            if field.location['x'] != -9999:
                field.send_keys(self.inn)
        session_id = self.browser.find_elements_by_css_selector('input[name=form_session_id]')
        return field_names

    def get_detail_link(self, fields):
        self.browser.find_element_by_css_selector('button[type=submit]').click()

        table = self.browser.find_element_by_css_selector('table.resultsTable')
        first_item = table.find_element_by_tag_name('a')

        return first_item.get_attribute('href')

    def get_detail_bo(self):
        self.browser.implicitly_wait(random.randint(3, 6))
        self.browser.get(self.linkDetailBO)
        self.browser.implicitly_wait(random.randint(3, 6))
        elem_a = self.browser.find_element_by_css_selector('.firmInfo a')

        a_link = elem_a.get_attribute('href')

        return a_link

    def check_company_status(self):
        self.browser.implicitly_wait(random.randint(3, 6))
        self.browser.get(self.linkDetailCheck)
        self.browser.implicitly_wait(random.randint(3, 6))
        result = ''

        elem_status = self.check_element_exist('.status-active')
        status = 0
        if elem_status is None:
            elem_status = self.check_element_exist('.status-liquidated')
            status = 1  # статус уже в любом случае Неактивна
        if elem_status is None:
            elem_status = self.check_element_exist('.status-bankruptcy')

        if elem_status is None:
            elem_status = self.check_element_exist('.status-reorganization')

        if elem_status is not None:
            result = elem_status.text

        date = ''
        if status == 1:
            date_elem = self.browser.find_element_by_css_selector('div.alert-paragraph a')
            date = date_elem.text

        age = None
        empNumber = None
        firmScale = None

        tableQuickInfo = self.check_element_exist('table.quick-profile')
        if tableQuickInfo:
            tableTr = tableQuickInfo.find_elements_by_css_selector('tr')

            for tr in tableTr:
                firstTd = tr.find_element_by_css_selector('td.quick-header')
                if 'Возраст' in firstTd.text:
                    age = tr.find_element_by_css_selector('td strong').text

                if 'Численность работников' in firstTd.text:
                    empNumber = tr.find_element_by_css_selector('td strong').text

        firmScaleBox = self.check_element_exist('div.firm-scale-box')
        if firmScaleBox:
            scaleFilled = firmScaleBox.find_elements_by_css_selector('.firm-scale-item.firm-scale-item-filled')
            firmScale = len(scaleFilled)

        return {
            'status': status,
            'statusText': result,
            'date': date,
            'age': age,
            'empNumber': empNumber,
            'firmScale': firmScale,
            'browser': self.browser
        }

    def check_element_exist(self, css_str):
        try:
            element = self.browser.find_element_by_css_selector(css_str)
        except NoSuchElementException:
            return None
        return element

