# fetch_web_corpus.py — Fixed version with reliable sources
import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin

WEB_RAW_DIR = "data/raw/web"
os.makedirs(WEB_RAW_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_text(text):
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and len(line) > 20]
    return "\n\n".join(lines)

def save_article(source, title, url, text):
    if len(text.strip()) < 300:
        return False
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:80]
    filename = f"{source}__{safe_title}.txt"
    filepath = os.path.join(WEB_RAW_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"SOURCE: {source}\n")
        f.write(f"TITLE: {title}\n")
        f.write(f"URL: {url}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)
    return True

def fetch_page(url, delay=1.0):
    try:
        time.sleep(delay)
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"    ✗ {url}: {e}")
        return None


# ── SOURCE 1: Senses of Cinema Great Directors (fixed) ───────────────────────

def crawl_senses_of_cinema():
    print("\n=== Senses of Cinema: Great Directors ===\n")
    saved = 0
    visited = set()

    # Their archive paginates — crawl multiple pages
    for page in range(1, 15):
        if page == 1:
            url = "https://www.sensesofcinema.com/category/great-directors/"
        else:
            url = f"https://www.sensesofcinema.com/category/great-directors/page/{page}/"

        print(f"  Index page {page}...")
        soup = fetch_page(url, delay=1.0)
        if not soup:
            break

        # Find article links on this index page
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (
                "sensesofcinema.com" in href and
                href not in visited and
                re.search(r'/\d{4}/', href) and  # has year in URL = article
                "great-directors" in href
            ):
                links.append(href)
                visited.add(href)

        if not links:
            print(f"  No more links found at page {page}, stopping.")
            break

        for article_url in links:
            soup2 = fetch_page(article_url, delay=0.8)
            if not soup2:
                continue

            title_tag = soup2.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "untitled"

            content = (
                soup2.find("div", class_="entry-content") or
                soup2.find("div", class_="post-content") or
                soup2.find("article")
            )
            if not content:
                continue

            text = clean_text(content.get_text(separator="\n"))
            if save_article("senses_of_cinema", title, article_url, text):
                saved += 1
                print(f"    ✓ {title[:70]}")

    print(f"\n  Senses of Cinema: {saved} saved")
    return saved


# ── SOURCE 2: Roger Ebert Great Movies ───────────────────────────────────────

def crawl_roger_ebert():
    """
    Crawls Roger Ebert's Great Movies essays.
    These are the gold standard for analytical film writing —
    long-form essays on canonical films with deep thematic analysis.
    """
    print("\n=== Roger Ebert: Great Movies ===\n")
    saved = 0
    visited = set()

    for page in range(1, 20):
        index_url = f"https://www.rogerebert.com/great-movies?page={page}"
        print(f"  Index page {page}...")
        soup = fetch_page(index_url, delay=1.0)
        if not soup:
            break

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/reviews/" in href and href not in visited:
                full_url = urljoin("https://www.rogerebert.com", href)
                links.append(full_url)
                visited.add(href)

        if not links:
            break

        for url in links:
            soup2 = fetch_page(url, delay=0.8)
            if not soup2:
                continue

            title_tag = soup2.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "untitled"

            content = (
                soup2.find("div", class_="review-content") or
                soup2.find("div", class_="entry-content") or
                soup2.find("article")
            )
            if not content:
                continue

            text = clean_text(content.get_text(separator="\n"))
            if save_article("roger_ebert", title, url, text):
                saved += 1
                print(f"    ✓ {title[:70]}")

    print(f"\n  Roger Ebert: {saved} saved")
    return saved


# ── SOURCE 3: Strictly Film School ───────────────────────────────────────────

def crawl_strictly_film_school():
    """
    Director database with 500+ analytical capsules.
    Auteurist focus, includes world and Indian cinema.
    """
    print("\n=== Strictly Film School: Directors ===\n")
    saved = 0
    visited = set()

    index_url = "http://www.filmref.com/directors/index.html"
    soup = fetch_page(index_url, delay=1.0)
    if not soup:
        # Try alternate URL
        index_url = "https://www.strictlyfilm.com/directors"
        soup = fetch_page(index_url, delay=1.0)

    if not soup:
        print("  ✗ Could not load index")
        return 0

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(index_url, href)
        if full_url not in visited and "director" in full_url.lower():
            links.append(full_url)
            visited.add(full_url)

    print(f"  Found {len(links)} director pages")

    for url in links:
        soup2 = fetch_page(url, delay=0.8)
        if not soup2:
            continue

        title_tag = soup2.find("h1") or soup2.find("h2")
        title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]

        body = soup2.find("body")
        if not body:
            continue

        text = clean_text(body.get_text(separator="\n"))
        if save_article("strictly_film_school", title, url, text):
            saved += 1
            print(f"  ✓ {title[:70]}")

    print(f"\n  Strictly Film School: {saved} saved")
    return saved


# ── SOURCE 4: Baradwaj Rangan (fixed) ────────────────────────────────────────

def crawl_baradwaj_rangan(max_pages=40):
    """
    Best analytical writing on Indian cinema anywhere on the web.
    """
    print("\n=== Baradwaj Rangan Blog ===\n")
    saved = 0
    visited = set()

    for page_num in range(1, max_pages + 1):
        index_url = (
            "https://baradwajrangan.wordpress.com/"
            if page_num == 1
            else f"https://baradwajrangan.wordpress.com/page/{page_num}/"
        )
        print(f"  Page {page_num}/{max_pages}...")
        soup = fetch_page(index_url, delay=1.0)
        if not soup:
            break

        post_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (
                "baradwajrangan.wordpress.com" in href and
                href not in visited and
                re.search(r'/\d{4}/\d{2}/\d{2}/', href)
            ):
                post_links.append(href)
                visited.add(href)

        if not post_links:
            print(f"  No posts found at page {page_num}, stopping.")
            break

        for url in post_links:
            soup2 = fetch_page(url, delay=0.8)
            if not soup2:
                continue

            title_tag = soup2.find("h1", class_="entry-title") or soup2.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "untitled"

            content = (
                soup2.find("div", class_="entry-content") or
                soup2.find("div", class_="post-content")
            )
            if not content:
                continue

            text = clean_text(content.get_text(separator="\n"))
            if save_article("baradwaj_rangan", title, url, text):
                saved += 1
                print(f"    ✓ {title[:70]}")

    print(f"\n  Baradwaj Rangan: {saved} saved")
    return saved


# ── SOURCE 5: Film Comment (public articles) ──────────────────────────────────

def crawl_film_comment(max_pages=15):
    """
    Film Comment — serious criticism since 1962.
    Many articles publicly accessible.
    """
    print("\n=== Film Comment ===\n")
    saved = 0
    visited = set()

    for page in range(1, max_pages + 1):
        index_url = f"https://www.filmcomment.com/articles/?page={page}"
        print(f"  Page {page}/{max_pages}...")
        soup = fetch_page(index_url, delay=1.2)
        if not soup:
            break

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (
                "/article/" in href and
                href not in visited
            ):
                full_url = urljoin("https://www.filmcomment.com", href)
                links.append(full_url)
                visited.add(href)

        for url in links:
            soup2 = fetch_page(url, delay=1.0)
            if not soup2:
                continue

            title_tag = soup2.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "untitled"

            content = (
                soup2.find("div", class_="article-content") or
                soup2.find("div", class_="entry-content") or
                soup2.find("article")
            )
            if not content:
                continue

            text = clean_text(content.get_text(separator="\n"))
            if save_article("film_comment", title, url, text):
                saved += 1
                print(f"    ✓ {title[:70]}")

    print(f"\n  Film Comment: {saved} saved")
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CineQuery Web Corpus Fetcher v2")
    print("=" * 60)

    total = 0
    total += crawl_senses_of_cinema()
    total += crawl_roger_ebert()
    total += crawl_baradwaj_rangan()
    total += crawl_film_comment()
    total += crawl_strictly_film_school()

    print("\n" + "=" * 60)
    print(f"TOTAL ARTICLES SAVED: {total}")
    print(f"Location: {WEB_RAW_DIR}/")
    print("Next: run build_index.py to re-embed everything")
    print("=" * 60)

if __name__ == "__main__":
    main()