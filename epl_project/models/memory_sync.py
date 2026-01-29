
import os
import json
from google import genai

def update_team_memory(team_name: str, api_key: str):
    """지식 그래프와 뉴스를 기반으로 팀의 전술 기억을 자동으로 업데이트 (KGGen 적용)"""
    kg_path = "epl_project/data/tactical_knowledge_graph.json"
    news_path = "epl_project/data/latest_epl_data.json"
    memory_path = "epl_project/data/team_memory.json"
    
    # 1. 관련 지식 수집
    relevant_facts = []
    if os.path.exists(kg_path):
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg = json.load(f)
            relevant_facts = [f"{t['subject']} {t['relation']} {t['object']}" for t in kg if team_name.lower() in t['subject'].lower() or team_name.lower() in t['object'].lower()]
            
    # 2. 뉴스 제목 수집
    relevant_news = []
    if os.path.exists(news_path):
        with open(news_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
            relevant_news = [n['title'] for n in news_data.get('news', []) if team_name.lower() in n['title'].lower()]

    if not relevant_facts and not relevant_news:
        return

    # 3. 브레인(Gemini)을 통한 전술 기억 증류
    client = genai.Client(api_key=api_key)
    prompt = f"""
    당신은 'Antigravity' 프로젝트의 30년 차 시니어 축구 분석가입니다.
    다음의 실시간 지식 그래프와 뉴스 데이터를 바탕으로 '{team_name}'의 전술 기억(Team Memory)을 업데이트하세요.
    맨유의 경우 최근 감독 교체(캐릭 등)와 경기 결과(맨시티전 승리 등)를 반드시 반영하세요.

    [지식 그래프 데이터]:
    {relevant_facts}

    [최신 뉴스 뉴스]:
    {relevant_news}

    [작성 가이드]:
    1. 'AI 시공간 대조 분석' 섹션을 포함하여 흐름 변화를 설명하세요.
    2. 전문 용어보다는 비유와 쉬운 해설을 섞어 '친절한 해설'로 작성하세요.
    3. 반드시 마크다운 형식을 유지하세요.
    """

    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        new_memory_text = response.text
        
        # 4. 저장
        memories = {}
        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memories = json.load(f)
        
        memories[team_name] = new_memory_text
        
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memories, f, ensure_ascii=False, indent=4)
        print(f"✅ {team_name} 전술 기억 업데이트 완료.")
    except Exception as e:
        print(f"❌ Memory Update Error: {e}")

if __name__ == "__main__":
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        update_team_memory("Manchester United", k)
        update_team_memory("Manchester City", k)
