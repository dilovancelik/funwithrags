from bs4 import BeautifulSoup
from selenium import webdriver
import time
from tqdm import tqdm
import json

BASE_URL = 'https://www.retsinformation.dk{}'
PAGE_URL = 'https://www.retsinformation.dk/documents?dt=10&h=false&ps=100&page={}'
PAGES = range(1,20)

driver = webdriver.Chrome()

def parse_string(titel, soup) -> str:
    results = []
    paragraphs = soup.find_all("p")
    heading = "intro"
    content = []
    grouper = ""
    if soup.find("p", attrs={"class": "CentreretParagraf"}):
        grouper = "CentreretParagraf"
    elif soup.find("p", attrs={"class": "ParagrafGruppeOverskrift"}):
        grouper = "ParagrafGruppeOverskrift"
    else:
        grouper = "Kapitel"
        
    for p in paragraphs:
        if p.find("p", attrs={"class": grouper}):
            results.append(f"{titel} ({heading}): {' '.join(content)}")
            heading = p.get_text().replace("\n", "")
            content = []
        elif p.find("p", attrs={"class": "Bilag" }):
            break
        elif p:
            content.append(p.get_text().replace("\n", " "))
    results.append(f"{titel} ({heading}): {' '.join(content)}")

    return results
processed_docs = []

with open("processed.csv", "r") as f:
    processed_docs = f.readlines()


with tqdm(total = 895) as pbar:
    for page in PAGES:
            url = PAGE_URL.format(page)
            
            driver.get(url)
            time.sleep(4)
            res = driver.page_source
            soup = BeautifulSoup(res, features='lxml')
            rows = soup.find("div", attrs={'class': 'search-result-list'})
            rows = rows.find_all("div", attrs={'typeof': 'eli:LegalResource'})
            for row in rows: 
                try: 
                    law_link = row.find_all("a", attrs={'class': 'document-title'})[0]
                    titel = law_link.get_text()
                    if (len(row.find_all("span", attrs={'resource': 'eli:InForce-inForce'})) == 0) or (f"{law_link["href"]}\n" in processed_docs) or (titel == "Lov om indfødsrets meddelelse"):
                        pbar.update(1)
                    else:
                        driver.get(BASE_URL.format(law_link['href']))
                        time.sleep(3)
                        law_soup = BeautifulSoup(driver.page_source, features='lxml')
                        law = law_soup.find("div", attrs={'class': 'document-content'})
                        law_documents = parse_string(titel, law)
                        with open("documents.csv", "a") as f:
                            for doc in law_documents:
                                f.write(f"{doc}\n")
                        with open("processed.csv", "a") as f:
                            f.write(f"{law_link['href']}\n")
                        pbar.update(1)
                except Exception as e:
                    with open("errors.csv", "a") as f:
                        f.write(f"{law_link['href']}\n")
                    raise e
    
