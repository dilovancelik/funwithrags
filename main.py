from bs4 import BeautifulSoup
from selenium import webdriver
import time
from tqdm import tqdm

BASE_URL = 'https://www.retsinformation.dk{}'
PAGE_URL = 'https://www.retsinformation.dk/documents?dt=10&ps=100&page={}'
PAGES = range(0,25)

driver = webdriver.Chrome()

with tqdm(total = 2441) as pbar:
    for page in PAGES:
        try: 
            url = PAGE_URL.format(page)
            
            driver.get(url)
            time.sleep(1)
            res = driver.page_source
            soup = BeautifulSoup(res, features='lxml')
            rows = soup.find("div", attrs={'class': 'search-result-list'})
            rows.find_all("div", attrs={'typeof': 'eli:LegalResource'})
            for row in rows: 
                if len(row.find_all("span", attrs={'resource': 'eli:InForce-inForce'})) == 0:
                    pbar.update(1)
                    continue
                law_link = row.find_all("a", attrs={'class': 'document-title'})[0]

                driver.get(BASE_URL.format(law_link['href']))
                time.sleep(2)
                law_soup = BeautifulSoup(driver.page_source)
                law = law_soup.find("div", attrs={'class': 'document-content'}).get_text()
                title = law_link['href'].replace('/', '-')[1:]
            
                file_path = f"data2/{title}.txt"

                with open(file_path, 'w') as f:
                    f.write(law)
                pbar.update(1)
        except:
            print(law_link['href'])
    
        
