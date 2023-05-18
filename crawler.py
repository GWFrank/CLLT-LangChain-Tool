import time
import re
import requests
from bs4 import BeautifulSoup

def pts_crawler(cnt):
    base_url = 'https://news.pts.org.tw/category/1'
    page = 1
    post_urls = set()
    posts = []
    url_reg = re.compile('https://news.pts.org.tw/article/[0-9]+')
    while True:
        page_res = requests.get(base_url, {'page': page})
        if page_res.status_code != requests.codes.ok:
            continue
        page_soup = BeautifulSoup(page_res.text, 'html')
        urls = [a.get('href') for a in page_soup.select('a') if a.get('href') != None]
        urls = set(url for url in urls if url_reg.match(url) != None)
        post_urls = post_urls.union(urls)
        if len(post_urls) >= cnt:
            break
        page += 1

    for post_url in list(post_urls)[:cnt]:
        post_res = requests.get(post_url, timeout=3)
        if post_res.status_code != requests.codes.ok:
            continue
        post_soup = BeautifulSoup(post_res.text, 'html')
        title = post_soup.select('h1', {'class': 'article-title'})[0].text
        title = post_soup.select('h1')[0].text.strip()
        datetime = post_soup.find_all(
                    'span', {'class': 'text-nowrap'}
                   )[0].text
        content = ''.join(p.text for p in post_soup.select('p'))
        editor = post_soup.find_all(
                    'span', {'class': 'article-reporter mr-2'}
                 )[0].text
        posts.append(
            {
                'title': title,
                'datetime': datetime,
                'editor': editor,
                'content': content,
            }
        )
    return posts

def ttv_crawler(cnt):
    base_url = 'https://news.ttv.com.tw/category/%E6%94%BF%E6%B2%BB'
    page = 1
    post_urls = set()
    posts = []
    while True:
        page_url = base_url + f'/{page}'
        page_res = requests.get(page_url, timeout=3)
        if page_res.status_code != requests.codes.ok:
            continue
        page_soup = BeautifulSoup(page_res.text, 'html')
        for tag in page_soup.select('main')[0].select('a'):
            post_urls.add(tag.get('href'))
            if len(post_urls) >= cnt:
                break
        if len(post_urls) >= cnt:
            break

    for post_url in list(post_urls)[:cnt]:
        post_res = requests.get(post_url, timeout=3)
        if post_res.status_code != requests.codes.ok:
            continue
        post_soup = BeautifulSoup(post_res.text, 'html')
        title = post_soup.select('h1')[0].text.strip()
        datetime = post_soup.find_all(
                    'li', {'class': 'date time'}
                   )[0].text.strip()
        content = ''.join(p.text for p in post_soup.select('p')[:-1])
        editor = post_soup.select('p')[-1].text
        posts.append(
            {
                'title': title,
                'datetime': datetime,
                'editor': editor,
                'content': content,
            }
        )
    return posts

news2crawler = {
    'pts': pts_crawler,
    'ttv': ttv_crawler,
}

def politic_news_crawler(news, cnt=10):
    if news in news2crawler.keys():
        news2crawler[news](cnt)
    else:
        raise NotImplementedError

