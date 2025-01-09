from bs4 import BeautifulSoup
import requests as rq
import urllib.parse
import os
import time

BASE_URL = 'https://danskelove.dk'
CHARACTERS = 'abcdefghijklmnopqrstuvwxyzæøå'

for c in CHARACTERS:
    url = f"{BASE_URL}/register/{c}"
    res = rq.get(url)
    soup = BeautifulSoup(res.text)
    laws = soup.find_all("ul", attrs={'class': None})[0]
    for law in laws.find_all('a', href=True):
        law_url = law["href"]
        title = urllib.parse.unquote(law_url, encoding="utf-8")
        title = title.replace("/","")
        file_path = f"data/{title}"
        if os.path.exists(file_path) or title == "retsplejeloven":
            continue
        time.sleep(5)
        law_res = rq.get(f"{BASE_URL}{law_url}")
        law_soup = BeautifulSoup(law_res.text)
        law_html = law_soup.find_all("div", {'id': 'content'})[0]
        with open(file_path, 'w') as f:
            f.write(law_html.get_text())

        print(f"wrote file to {file_path}")
   
