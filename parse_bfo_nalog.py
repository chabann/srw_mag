import requests
import json
from desctop_agents import get_header
import re


class FinancialCompanyData:
    """ Класс для взаимодействия с данными сайта https://bo.nalog.ru и анализа финансовой отчетности предприятий """

    def __init__(self, query='', inn='', name='', ogrn='', page='0'):
        self.urlSearch = 'https://bo.nalog.ru/nbo/organizations/search?query=#QUERY#&page=#PAGE#'  # url страницы
        self.urlGetReport = 'https://bo.nalog.ru/nbo/organizations/#COMPANY_ID#/bfo/'  # url для получения отчетов
        self.urlGetDetailReport = 'https://bo.nalog.ru/nbo/bfo/#REPORT_ID#/details'  # url для детального отчета
        self.query = query
        self.inn = inn
        self.name = name
        self.ogrn = ogrn
        self.page = page

        self.companyData = {}
        self.additCompanyData = {}
        self.detail_reports = []

        self.send_search_request()

    def send_search_request(self):
        """ Метод отправляет запрос по введенным данным для поиска компании """
        if self.inn:
            url_search = self.urlSearch.replace('#QUERY#', self.inn + '&allFieldsMatch=false&inn=' + self.inn)
        else:
            url_search = self.urlSearch.replace('#QUERY#', self.query)

        url_search = url_search.replace('#PAGE#', self.page)
        result = requests.get(url_search, headers=get_header())  # отправляем HTTP запрос
        try:
            take_company = json.loads(result.text)['content'][0]
            legal_id = take_company['id']  # пока что берем первую из найденных
            self.detail_reports = self.send_reports_request(legal_id)
            self.additCompanyData['name'] = take_company['shortName']
            self.additCompanyData['okved_id'] = take_company['okved2']['id']
            self.additCompanyData['okved_name'] = take_company['okved2']['name']
            self.additCompanyData['inn'] = re.sub(r'[^0-9]', '', take_company['inn'])
        except IndexError:
            print('Не найдено данных о компании ')
            print(json.loads(result.text))

    def send_reports_request(self, legal_id):
        """ Метод отправляет запрос для получения списка доступных отчетов по id компании """

        url_get_report = self.urlGetReport.replace('#COMPANY_ID#', str(legal_id))
        result = requests.get(url_get_report, headers=get_header())  # отправляем HTTP запрос

        return json.loads(result.text)

    def send_detail_request(self, report_id):
        """ Метод отправляет запрос на получение детальной финансовой отчетности по выбранному id отчета  """

        url_get_detail_report = self.urlGetDetailReport.replace('#REPORT_ID#', str(report_id))
        result = requests.get(url_get_detail_report, headers=get_header())  # отправляем HTTP запрос
        self.companyData = json.loads(result.text)[0]
