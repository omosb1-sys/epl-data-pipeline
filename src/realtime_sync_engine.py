import os
import requests
from bs4 import BeautifulSoup
import mysql.connector
from datetime import datetime
import re

def connect_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password=os.getenv("MYSQL_PASSWORD", ""),
            database="epl_x_db",
            auth_plugin='mysql_native_password'
        )
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return None

from custom_football_mcp import run_custom_mcp_sync

def scrape_news_sources():
    """다양한 소스에서 프리미어리그 소식을 수집합니다."""
    sources = {
        "BBC Sport": "https://www.bbc.com/sport/football/premier-league"
    }
    
    all_news = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    # [1] 기존 소스 크롤링 (BBC Only)
    for name, url in sources.items():
        try:
            print(f"📡 {name}에서 소식 수집 중...")
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # BBC 구조: h3 태그
            items = soup.find_all('h3', limit=10)
                
            for item in items:
                text = item.get_text().strip()
                if text and len(text) > 20: 
                    # BBC/Sky URL Extraction (Simple)
                    link = item.find_parent('a')['href'] if item.find_parent('a') else "#"
                    if link.startswith('/'):
                        base = "https://www.bbc.com" if "bbc" in url else "https://www.skysports.com"
                        link = base + link
                        
                    all_news.append({"source": name, "title": text, "url": link})
                    
        except Exception as e:
            print(f"❌ {name} 크롤링 중 오류: {e}")
            
    # [2] Custom MCP Integration
    try:
        print("🔌 Custom MCP (네이버 카페/전문 사이트) 가동 중...")
        mcp_news = run_custom_mcp_sync()
        all_news.extend(mcp_news)
    except Exception as e:
        print(f"❌ Custom MCP 실행 실패: {e}")

    # Remove duplicates (dict cannot be hashed, so specialized dedup)
    unique_news = []
    seen = set()
    for n in all_news:
        if isinstance(n, dict):
            key = n['title']
            if key not in seen:
                seen.add(key)
                unique_news.append(n)
        else:
             # Legacy string support? No, convert all to dict
             pass

    return unique_news

def auto_update_db_from_news(news_list):
    """수집된 뉴스 텍스트를 분석하여 DB에 자동 반영합니다."""
    conn = connect_db()
    if not conn: return
    cursor = conn.cursor()
    
    # 1. 키워드 기반 자동 업데이트 로직 (간단한 AI 매칭 예시)
    # 실제로는 LLM이나 NLP를 쓰면 더 정확하지만, 여기선 키워드 규칙 기반으로 구현
    
    updates_made = []
    
    # 팀 이름 한글-영문 매핑 (매칭을 위해)
    team_keywords = {
        "아스날": ["Arsenal", "Arteta", "Gyokeres", "Calafiori"],
        "맨체스터 유나이티드": ["United", "Man Utd", "Amorim", "Sesko", "Yoro"],
        "리버풀": ["Liverpool", "Salah", "Slot", "Bradley"],
        "맨체스터 시티": ["City", "Guardiola", "Haaland", "Rodri", "Stones"],
        "토트넘 홋스퍼": ["Tottenham", "Spurs", "Son", "Maddison", "Vicario"],
        "첼시": ["Chelsea", "Maresca", "Colwill", "Lavia"]
    }

    for news_item in news_list:
        # Data structure check
        if isinstance(news_item, dict):
            news_text = news_item.get('title', "")
        else:
            news_text = str(news_item)
            
        # 부상 관련 키워드 감지
        if any(w in news_text.lower() for w in ["injury", "sidelined", "out for", "blow", "hurt"]):
            for team, keys in team_keywords.items():
                if any(k.lower() in news_text.lower() for k in keys):
                    cursor.execute("UPDATE clubs SET injury_level = '심각' WHERE team_name LIKE %s", (f"%{team}%",))
                    updates_made.append(f"🚑 {team} 부상 소식 자동 반영")

        # 이적 관련 키워드 감지
        if any(w in news_text.lower() for w in ["signs", "agreement", "deal", "official", "confirmed"]):
            for team, keys in team_keywords.items():
                if any(k.lower() in news_text.lower() for k in keys):
                    # 여기에 실제 선수 이름을 파싱하는 고도의 로직이 들어가면 좋음
                    # 지금은 데이터 정합성 유지 알림만 기록
                    updates_made.append(f"🔄 {team} 이적/계약 소식 감지됨")

    conn.commit()
    conn.close()
    return updates_made

def sync_data():
    """외부 소스 동기화 및 DB 자동 보충 메인 함수"""
    news = scrape_news_sources()
    updates = auto_update_db_from_news(news)
    
    # 결과 요약 반환
    return {
        "news": news,
        "updates": updates,
        "timestamp": datetime.now().strftime('%H:%M:%S')
    }

if __name__ == "__main__":
    result = sync_data()
    print(f"✅ 동기화 완료! 뉴스 {len(result['news'])}건 수집, {len(result['updates'])}건 자동 반영.")
