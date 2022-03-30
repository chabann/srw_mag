import requests
import json
from desctop_agents import get_header


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

        self.send_search_request()

    def send_search_request(self):
        """ Метод отправляет запрос по введенным данным для поиска компании """

        url_search = self.urlSearch.replace('#QUERY#', self.query).replace('#PAGE#', self.page)
        result = requests.get(url_search, headers=get_header())  # отправляем HTTP запрос
        legal_id = json.loads(result.text)['content'][0]['id']  # ToDo: пока что берем первую из найденных

        self.send_reports_request(legal_id)

    def send_reports_request(self, legal_id):
        """ Метод отправляет запрос для получения списка доступных отчетов по id компании """

        url_get_report = self.urlGetReport.replace('#COMPANY_ID#', str(legal_id))
        result = requests.get(url_get_report, headers=get_header())  # отправляем HTTP запрос
        report_id = json.loads(result.text)[0]['id']  # берем последний из найденных отчетов

        self.send_detail_request(report_id)

    def send_detail_request(self, report_id):
        """ Метод отправляет запрос на получение детальной финансовой отчетности по выбранному id отчета  """

        url_get_detail_report = self.urlGetDetailReport.replace('#REPORT_ID#', str(report_id))
        result = requests.get(url_get_detail_report, headers=get_header())  # отправляем HTTP запрос
        self.companyData = json.loads(result.text)[0]
