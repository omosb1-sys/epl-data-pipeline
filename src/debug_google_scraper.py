
import requests
from bs4 import BeautifulSoup

def debug_google_scrape():
    query = "맨유"
    # Same params as current code
    url = f"https://www.google.com/search?q={query}&tbm=nws&gl=kr&hl=ko"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
    }
    
    print(f"Fetching: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check current selector
        elements = soup.select('div.BNeawe.vvjwJb.AP7Wnd')
        print(f"Found {len(elements)} elements with selector 'div.BNeawe.vvjwJb.AP7Wnd'")
        
        if not elements:
            print("Trying fallback selectors...")
            # Try finding any standard search results container
            links = soup.find_all('a')
            print(f"Total links found: {len(links)}")
            for i, a in enumerate(links[:20]):
                print(f"Link {i}: {a.get('href')} - Text: {a.get_text()[:30]}")
                
        for el in elements[:3]:
            title = el.get_text().strip()
            parent_a = el.find_parent('a')
            link = "No parent 'a' tag"
            if parent_a:
                link = parent_a.get('href')
            print(f"Title: {title}")
            print(f"Raw Link: {link}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_google_scrape()
