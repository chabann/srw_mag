import requests
import json
from desctop_agents import get_header


class MainCompanyData:
    """ Класс для взаимодействия с данными сайта https://pb.nalog.ru """

    def __init__(self, inn='', query='', page='0'):
        self.urlSearch = 'https://pb.nalog.ru/search-proc.json?mode=search-all&queryAll=#INN#'
        self.urlGetTokenAndId = 'https://pb.nalog.ru/company-proc.json?token=#TOKEN#&method=get-request'
        self.urlDetail = 'https://pb.nalog.ru/company-proc.json?token=#TOKEN#&method=get-response&id=#ID#'
        self.query = query
        self.inn = inn
        self.page = page

        self.token = ''
        self.id = ''
        self.companyData = {}
        self.detailResult = {}

        self.positiveStatuses = ['действующее', 'находится в процессе реорганизации в форме слияния',
                                 'находится в процессе реорганизации в форме преобразования',
                                 'находится в процессе реорганизации в форме присоединения к нему других ЮЛ',
                                 'находится в процессе реорганизации в форме присоединения к другому ЮЛ',
                                 'прекратило деятельность в связи с приобретением главы КФХ статуса ИП',
                                 'регдело передано в другой регорган']

        self.negativeStatuses = ['исключен из ЕГРЮЛ', 'ликвидировано', 'ликвидировано вследствие банкротства',
                                 'ликвидировано по решению суда',
                                 'некоммерческая организация ликвидирована по решению суда',
                                 'находится в стадии ликвидации',
                                 'находится в процессе реорганизации в форме присоединения '
                                 '(прекращает деятельность после реорганизации)',
                                 'прекратило деятельность при присоединении',
                                 'прекратило деятельность при преобразовании',
                                 'прекратило деятельность',
                                 'прекратило деятельность при слиянии',
                                 'принято решение о предстоящем исключении недействующего ЮЛ из ЕГРЮЛ',
                                 'регистрация признана недействительной по решению суда']

    def send_to_detail(self):
        url_search = self.urlSearch.replace('#INN#', str(self.inn))
        result = requests.get(url_search, headers=get_header())
        if len(json.loads(result.text)['ul']['data']):
            token = json.loads(result.text)['ul']['data'][0]['token']

            url_search = self.urlGetTokenAndId.replace('#TOKEN#', token)
            result = requests.get(url_search, headers=get_header())

            self.id = json.loads(result.text)['id']
            self.token = json.loads(result.text)['token']

            url_search = self.urlDetail.replace('#TOKEN#', self.token).replace('#ID#', self.id)
            self.detailResult = json.loads(requests.get(url_search, headers=get_header()).text)

    def get_company_status(self):
        self.send_to_detail()

        if (self.token != '') and (self.id != ''):
            # 1 - банкрот, 0 - не банкрот
            if ('НаимСтатусЮЛСокр' in self.detailResult['vyp'])\
                    or ('НаимСтатусЮЛ' in self.detailResult['vyp']):
                status = self.detailResult['vyp']['НаимСтатусЮЛСокр']
                for st in self.positiveStatuses:
                    if status == st:
                        return 0
                return 1
            else:
                return 0
        else:
            return 1
