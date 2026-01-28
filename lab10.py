import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
from urllib.parse import urljoin
import time

BASE_URL = "https://pl.wikipedia.org"
HOME_URL = f"{BASE_URL}/wiki/Wikipedia:Strona_główna"

HEADERS = {
    "User-Agent": "wiki-scraper/1.0 (educational purpose)"
}


def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.text


def parse_article(url):
    html = fetch(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    title = soup.find("h1")
    title = title.get_text(strip=True) if title else "Brak tytułu"

    paragraph = ""
    for p in soup.select("div.mw-parser-output > p"):
        if p.get_text(strip=True):
            paragraph = p.get_text(strip=True)
            break

    return {
        "url": url,
        "title": title,
        "intro": paragraph
    }

def get_links_from_homepage():
    html = fetch(HOME_URL)
    soup = BeautifulSoup(html, "html.parser")

    links = set()

    for a in soup.select("a[href^='/wiki/']"):
        href = a.get("href")

        if ":" in href:
            continue

        full_url = urljoin(BASE_URL, href)
        links.add(full_url)

    return list(links)

if __name__ == "__main__":
    start = time.time()

    links = get_links_from_homepage()

    print(f"Znaleziono {len(links)} linków\n")

    # workers = min(cpu_count()-1, 8) 
    workers = 1

    with Pool(processes=workers) as pool:
        results = pool.map(parse_article, links[:40])  

    results = [r for r in results if r]

    print(f"Pobrano {len(results)} artykułów:\n")

    for r in results[:5]:
        print("Tytuł:", r["title"])
        print("URL:", r["url"])
        print("Intro:", r["intro"][:200], "...\n")

    print(f"Czas wykonania: {time.time() - start:.2f}s")