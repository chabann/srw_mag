import requests
from bs4 import BeautifulSoup
import pandas as pd
from random import choice


def random_headers():
    return {'User-Agent': choice(desktop_agents), 'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}


def parse_table(review, res):  # Функция разбора таблицы с вопросом
    titleblock = review.find('h2')
    if titleblock != -1:
        title = titleblock.find('a').text

        rate = review.find('span', {'class': 'star_ring_big'}).get('title').replace('Оценка автора: ', '')

        comment = review.find('span', {'class': 'description'}).text

        advantblock = review.find('div', {'class': 'advantages'})
        if advantblock != None:
            advant = advantblock.find('ol').text
        else:
            advant = ''

        disadvantblock = review.find('div', {'class': 'disadvantages'})
        if disadvantblock != None:
            disadvant = disadvantblock.find('ol').text
        else:
            disadvant = ''

        res = res.append(pd.DataFrame([[title, rate, comment, advant, disadvant]], columns=['title', 'rate', 'comment',
            'advant', 'disadvant']), ignore_index=True)
    return res


desktop_agents = ['Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
                 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
                 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
                 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
                 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
                 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']


companyName = 'internet-provayder_rostelekom_russia'

url = 'https://ru.otzyv.com/rostelekom'  # url страницы
result = pd.DataFrame()

r = requests.get(url, headers=random_headers())  # отправляем HTTP запрос и получаем результат
soup = BeautifulSoup(r.text, 'html.parser')  # Отправляем полученную страницу в библиотеку для парсинга

commonRating = soup.find('div', {'class': 'b_rate'}).text  # Общий рейтинг
print(commonRating)

reviewCountBlock = soup.find('div', {'class': 'otzyv_cnt'})  # количество отзывов
reviewCount = reviewCountBlock.find('span', {'class': 'count'}).text
print(reviewCount)

# result.append(pd.DataFrame([[commonRating, reviewCount]], columns=['commonRating', 'reviewCount']), ignore_index=True)

reviews = soup.find_all('div', {'class': 'comment_row'})  # Получаем все блоки с отзывами

for item in reviews:
    titleblock = item.find('h2')
    if titleblock != None:
        result = parse_table(item, result)

result.to_excel('data/resultTest.xlsx')

print(result)
