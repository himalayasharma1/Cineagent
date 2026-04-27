# fetch_corpus.py — uses Wikipedia REST API directly (more reliable)

import requests
import os
import time

DIRECTORS = [
    "Stanley Kubrick", "Martin Scorsese", "Christopher Nolan",
    "David Fincher", "Paul Thomas Anderson", "Coen Brothers",
    "Wes Anderson", "Denis Villeneuve", "Quentin Tarantino",
    "Alfonso Cuarón", "Steven Spielberg", "Spike Lee",
    "Sidney Lumet", "David Lynch", "Terrence Malick",
    "Ridley Scott", "James Cameron", "Peter Weir",
    "Richard Linklater", "Akira Kurosawa", "Satyajit Ray",
    "Shyam Benegal", "Mrinal Sen", "Mani Ratnam",
    "Anurag Kashyap", "Imtiaz Ali", "Vishal Bhardwaj",
    "Rituparno Ghosh", "Guru Dutt",
]

ACTORS = [
    "Leonardo DiCaprio", "Al Pacino", "Marlon Brando",
    "Robert De Niro", "Ethan Hawke", "Dev Anand",
    "Irrfan Khan", "Brad Pitt", "Shah Rukh Khan",
]

def fetch_wikipedia(name):
    """
    Fetch plain text content from Wikipedia using the REST API.
    More reliable than the wikipedia Python library.
    """
    # Step 1: Search for the best matching page title
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "format": "json",
        "srlimit": 1,
    }
    
    headers = {"User-Agent": "CineQuery/1.0 (educational project)"}
    
    search_resp = requests.get(search_url, params=search_params, headers=headers, timeout=10)
    search_data = search_resp.json()
    
    results = search_data.get("query", {}).get("search", [])
    if not results:
        return None, "No search results found"
    
    page_title = results[0]["title"]
    
    # Step 2: Fetch the full plain text of that page
    extract_params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,       # plain text, no HTML
        "exsectionformat": "plain",
        "titles": page_title,
        "format": "json",
    }
    
    extract_resp = requests.get(search_url, params=extract_params, headers=headers, timeout=10)
    extract_data = extract_resp.json()
    
    pages = extract_data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    content = page.get("extract", "")
    
    if not content:
        return None, "Empty content returned"
    
    return content, page_title


def save_entry(name, category, content, page_title, output_dir):
    safe_name = name.replace(" ", "_").replace("/", "_")
    filename = f"{category}_{safe_name}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"CATEGORY: {category.upper()}\n")
        f.write(f"SUBJECT: {name}\n")
        f.write(f"WIKIPEDIA TITLE: {page_title}\n")
        f.write(f"SOURCE: Wikipedia\n")
        f.write("=" * 60 + "\n\n")
        f.write(content)
    
    return filename, len(content)


def process_list(names, category, output_dir):
    results = []
    for name in names:
        print(f"Fetching: {name}...")
        try:
            content, page_title = fetch_wikipedia(name)
            if content:
                filename, chars = save_entry(name, category, content, page_title, output_dir)
                print(f"  ✓ Saved: {filename} ({chars:,} characters)")
                results.append((name, True))
            else:
                print(f"  ✗ Failed: {page_title}")
                results.append((name, False))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append((name, False))
        
        time.sleep(0.3)  # polite delay between requests
    
    return results


def main():
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Fetching Director Pages ===\n")
    director_results = process_list(DIRECTORS, "director", output_dir)

    print("\n=== Fetching Actor Pages ===\n")
    actor_results = process_list(ACTORS, "actor", output_dir)

    print("\n=== Summary ===\n")
    all_results = director_results + actor_results
    succeeded = [n for n, ok in all_results if ok]
    failed = [n for n, ok in all_results if not ok]

    print(f"✓ Successfully fetched: {len(succeeded)}/{len(all_results)}")
    if failed:
        print(f"✗ Failed: {', '.join(failed)}")
    print(f"\nFiles saved to: {output_dir}/")


if __name__ == "__main__":
    main()