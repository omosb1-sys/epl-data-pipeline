
import os
import time
import feedparser
from datetime import datetime
from google import genai

# 1. 설정
RSS_URL = "https://news.hada.io/rss/news"
SAVE_DIR = "research/readings/geeknews_trends"
os.makedirs(SAVE_DIR, exist_ok=True)

api_key = os.environ.get("GEMINI_API_KEY")

def get_ai_wisdom(title, desc):
    """Geek News 기사를 분석하여 Antigravity 프로젝트 적용점 도출 (Gemini 3)"""
    if not api_key:
        return "💡 분석 대기: API Key 설정이 필요합니다."
    
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        당신은 'Antigravity' 프로젝트의 시니어 데이터 분석가입니다. 
        개발자 및 데이터 과학자 커뮤니티인 Geek News의 다음 소식을 분석하세요.
        
        [뉴스 제목]: {title}
        [뉴스 요약]: {desc[:600]}
        
        [미션]:
        이 소식이 우리의 축구 데이터 분석(EPL/K-League), 예측 모델, 또는 파이썬 엔지니어링 효율성에 줄 수 있는 
        핵심 인사이 트를 1줄로 도출하세요. 불필요한 서술은 제외하고 바로 인사이 트만 제시하세요.
        """
        
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        return response.text.replace('\n', ' ').strip()
    except Exception as e:
        return f"분석 대기 중... ({str(e)[:50]})"

def analyze_and_report(post):
    print(f"   ↳ 🧠 Geek 인사이트 추출 중... ", end=" ", flush=True)
    wisdom = get_ai_wisdom(post.title, post.summary if hasattr(post, 'summary') else post.description)
    print("✅")
    return f"### {post.title}\n\n**Data Analyst Insight:**\n{wisdom}\n\n**Link**: {post.link}"

def main():
    print("🤓 Geek News Knowledge Harvester (Intelligence Mode)")
    feed = feedparser.parse(RSS_URL)
    
    # 최신 뉴스 10개 분석
    posts = feed.entries[:10]
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    results = []
    for i, post in enumerate(posts):
        print(f"[{i+1}/{len(posts)}] {post.title[:45]}...")
        results.append(analyze_and_report(post))
        time.sleep(0.5)
        
    report_path = os.path.join(SAVE_DIR, f"geeknews_digest_{today_str}.md")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(f"# Geek News Daily Intelligence ({today_str})\n\n" + "\n\n".join(results))
        f.write(f"\n\n---\n*Insight Clean-up Routine: 적용 후 즉시 삭제 예정*")
        
    print(f"\n✨ Geek News 리포트 생성 완료: {report_path}")

if __name__ == "__main__":
    main()
