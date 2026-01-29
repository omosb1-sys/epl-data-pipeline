
import os
import time
import feedparser
from datetime import datetime
from google import genai

# 1. 설정
RSS_URL = "https://discuss.pytorch.kr/latest.rss"
SAVE_DIR = "research/readings/pytorch_trends"
os.makedirs(SAVE_DIR, exist_ok=True)

api_key = os.environ.get("GEMINI_API_KEY")

def get_ai_wisdom(title, desc):
    """지능적인 분석을 위한 제미나이 3 호출 (Data Analyst Perspective)"""
    if not api_key:
        return "💡 인사이결: AI 분석 엔진이 대기 중입니다."
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        당신은 30년 차 시니어 데이터 분석가입니다. 
        다음 뉴스/포스트를 분석하여 'Antigravity' AI 시스템에 적용할 수 있는 핵심 인사이트를 1줄로 도출하세요.
        축구 데이터 분석이나 에이전트 성능 개선 관점에서 생각하세요.
        
        제목: {title}
        내용: {desc[:500]}
        """
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        return response.text.replace('\n', ' ').strip()
    except Exception as e:
        return f"분석 대기 중... ({str(e)[:40]})"

def analyze_and_learn(post):
    print(f"   ↳ 🧠 AI 지식 추출 중...  ", end=" ", flush=True)
    wisdom = get_ai_wisdom(post.title, post.description)
    print("✅ 완료!")
    return f"### {post.title}\n\n**Data Analyst Insight:**\n{wisdom}\n\n**Link**: {post.link}"

def main():
    print("🤖 PyTorch Knowledge Harvester (Next-Gen Gemini 3 Mode)")
    feed = feedparser.parse(RSS_URL)
    posts = feed.entries[:10]
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    results = []
    for i, post in enumerate(posts):
        print(f"[{i+1}/{len(posts)}] {post.title[:40]}...")
        results.append(analyze_and_learn(post))
        time.sleep(0.5) # 부하 방지
        
    report_path = os.path.join(SAVE_DIR, f"digest_{today_str}.md")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(f"# PyTorch Daily Content ({today_str})\n\n" + "\n\n".join(results))
    print(f"\n✨ 리포트 생성 완료: {report_path}")

if __name__ == "__main__":
    main()
