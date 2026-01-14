
import requests
from bs4 import BeautifulSoup
import random
import time
import urllib.parse

def scrape_naver_cafe_manutd():
    # Naver Cafe (Mobile)
    url = "https://m.cafe.naver.com/as6060"
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
    }
    news_items = []
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        articles = soup.select('ul.list_area > li > a.txt_area')
        if not articles:
            articles = soup.select('.board_box') # Fallback
            
        for art in articles[:5]:
            try:
                # Title
                title_tag = art.select_one('strong.tit')
                title = title_tag.get_text().strip() if title_tag else "제목 없음"
                
                # Link
                # Mobile links in Naver Cafe list are often relative or js-based, but 'href' usually exists or is constructed.
                # In m.cafe.naver.com, href often looks like /SectionArticleRead.nhn?articleid=...
                link = art.get('href')
                if link and not link.startswith('http'):
                    link = f"https://m.cafe.naver.com{link}"
                
                news_items.append({"source": "맨유 카페", "title": title, "url": link})
            except:
                continue
                
    except Exception as e:
        news_items.append({"source": "System", "title": f"맨유 카페 접속 실패: {e}", "url": "#"})
        
    return news_items

def scrape_statsbomb():
    url = "https://statsbomb.com/articles/"
    headers = {"User-Agent": "Mozilla/5.0"}
    news_items = []
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Statsbomb (redirects to Hudl Blog now)
        # Proven selector: .card__title checks text, and parent <a> contains link
        titles = soup.select('.card__title')
        
        for t in titles[:5]:
            title = t.get_text().strip()
            # Link is usually in the parent <a> tag for these cards
            parent_a = t.find_parent('a')
            if parent_a:
                link = parent_a.get('href')
                # Hudl links are usually absolute, but check just in case
                if link and not link.startswith('http'):
                     link = f"https://www.hudl.com{link}"
                
                news_items.append({"source": "StatsBomb", "title": title, "url": link})
            else:
                # Fallback: check if the title itself is an <a> tag
                link_tag = t.find('a')
                if link_tag:
                     link = link_tag.get('href')
                     news_items.append({"source": "StatsBomb", "title": title, "url": link})
            
    except Exception as e:
        news_items.append({"source": "System", "title": f"StatsBomb 접속 실패: {e}", "url": "#"})
        
    return news_items


def scrape_google_news(query="프리미어리그 최신 뉴스", lang="ko"):
    # Dynamic Language Support
    if lang == "ko":
        gl_param = "kr"
        hl_param = "ko"
        accept_lang = "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
    else:
        gl_param = "us" # or uk
        hl_param = "en"
        accept_lang = "en-US,en;q=0.9"

    # Using 'tbm=nws' to ensure we get news results
    url = f"https://www.google.com/search?q={query}&tbm=nws&gl={gl_param}&hl={hl_param}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Accept-Language": accept_lang
    }
    news_items = []
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Based on debug output: links are like /url?esrc=s&q=&rct=j&sa=U&url=https://...
        # We process all 'a' tags to find these
        links = soup.find_all('a')
        
        count = 0
        for a in links:
            if count >= 5: break
            
            href = a.get('href')
            if href and href.startswith('/url?'):
                # Extract real URL
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                if 'url' in parsed:
                    real_url = parsed['url'][0]
                    
                    # Filter out internal Google links
                    if 'google.com' in real_url: continue
                    
                    # Extract Title
                    title_div = a.find('div', class_='BNeawe vvjwJb AP7Wnd')
                    if title_div:
                        title = title_div.get_text().strip()
                    else:
                        title = a.get_text().strip()
                        
                    if title and len(title) > 5: 
                        news_items.append({"source": "Google 뉴스", "title": title, "url": real_url})
                        count += 1
                        
    except Exception as e:
        news_items.append({"source": "System", "title": f"Google 검색 실패({query}): {e}", "url": "#"})
    
    return news_items

def scrape_insiders():
    """Fabrizio Romano & David Ornstein News"""
    data = []
    # Fabrizio Romano
    romano = scrape_google_news("Fabrizio Romano transfer", lang="en")
    for n in romano:
        n['source'] = "Fabrizio Romano"
        n['title'] = f"[Romano] {n['title']}"
    data.extend(romano)

    # David Ornstein
    ornstein = scrape_google_news("David Ornstein exclusive", lang="en")
    for n in ornstein:
        n['source'] = "David Ornstein"
        n['title'] = f"[Ornstein] {n['title']}"
    data.extend(ornstein)
    
    return data



def scrape_sky_sports():
    # Sky Sports Football News
    url = "https://www.skysports.com/football/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    news_items = []
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Sky Sports News List usually in 'a.news-list__headline-link'
        articles = soup.select('a.news-list__headline-link')
        
        for art in articles[:7]:
            title = art.get_text().strip()
            link = art.get('href')
            if link and not link.startswith('http'):
                link = f"https://www.skysports.com{link}"
                
            if title:
                news_items.append({"source": "Sky Sports", "title": title, "url": link})
                
    except Exception as e:
         news_items.append({"source": "System", "title": f"Sky Sports 접속 실패: {e}", "url": "#"})
         
    return news_items

def run_custom_mcp_sync():
    all_data = []
    print("running custom mcp...")
    
    # 1. Sky Sports (Major Media)
    try:
        all_data.extend(scrape_sky_sports())
    except: pass
    
    # 2. Specialized Sites
    all_data.extend(scrape_naver_cafe_manutd())
    all_data.extend(scrape_statsbomb())
    # all_data.extend(scrape_overlyzer()) # Removed
    
    # 3. Insiders (Romano/Ornstein)
    try:
        all_data.extend(scrape_insiders())
    except: pass
    
    # 4. Random Team Search (Google Korean News)
    pl_teams = [
        "애스턴 빌라", "리버풀", "아스널", "브라이튼", "맨체스터 시티", "첼시", 
        "뉴캐슬", "노팅엄", "토트넘", "브렌트포드", "울버햄튼", 
        "맨체스터 유나이티드", "웨스트햄", "본머스", "풀럼", "레스터 시티", 
        "크리스탈 팰리스", "에버턴", "입스위치", "사우스햄튼"
    ]
    
    target_teams = random.sample(pl_teams, 2) # Reduce to 2 to save time
    target_teams.append("프리미어리그")
    
    print(f"🔎 Monitoring: {target_teams}")
    
    for team in target_teams:
        try:
            # News Search
            query = f"{team} 최근 뉴스"
            news = scrape_google_news(query)
            # Add team tag
            for n in news:
                n['source'] = f"Google ({team})"
            all_data.extend(news)
            time.sleep(1)
        except:
            pass
            
    return all_data

    return all_data

def analyze_team_realtime(team_name):
    """
    Analyzes real-time news for a specific team to determine impact score.
    Returns: (score, summary)
    Score range: -5 (Bad) to +5 (Good)
    """
    # 1. Search Query (Korean + English Keywords for broader coverage)
    # We use English team names for better global coverage if possible, but keep it simple first
    query = f"{team_name} (injury OR transfer OR sack OR news)"
    
    # Use our existing scraper (defaulting to Korean results for user readability, 
    # but we might want English for analysis. Mixed approach: Search Korean High Priority)
    news = scrape_google_news(f"{team_name} 주요 뉴스", lang="ko")
    
    impact_score = 0
    summary_reasons = []
    
    # Keywords (Simple Sentiment Analysis)
    neg_keywords = ['부상', '결장', '경질', '불화', '패배', '위기', 'Injury', 'Out', 'Sack', 'Crisis']
    pos_keywords = ['영입', '복귀', '승리', '재계약', '부활', 'Return', 'Sign', 'Win', 'Fit']
    
    for n in news[:5]: # Analyze top 5 news
        title = n['title']
        
        # Check Negative
        for kw in neg_keywords:
            if kw in title:
                impact_score -= 2
                summary_reasons.append(f"📉 악재 감지: {title}")
                break # Count once per article
                
        # Check Positive
        for kw in pos_keywords:
            if kw in title:
                impact_score += 1 # Positives usually have less immediate impact on probability than injuries
                summary_reasons.append(f"📈 호재 감지: {title}")
                break

    # Cap Score
    impact_score = max(-10, min(10, impact_score))
    
    # Generate Summary
    if not summary_reasons:
        summary_text = "특이 사항 없음"
        impact_score = 0 # Neutral
    else:
        summary_text = " / ".join(summary_reasons[:3])
        
    return impact_score, summary_text, news

if __name__ == "__main__":
    print(run_custom_mcp_sync())
