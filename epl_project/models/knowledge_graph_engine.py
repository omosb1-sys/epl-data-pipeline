
import json
import os
from google import genai
from typing import List, Dict

class FootballKGEngine:
    """
    KGGen 방법론을 EPL 데이터 분석에 적용한 지식 그래프 엔진.
    비구조화된 뉴스/리포트에서 [엔티티 - 관계 - 엔티티]를 추출하여 전술 자산화함.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.graph_file = "epl_project/data/tactical_knowledge_graph.json"
        
    def extract_graph_from_text(self, text_list: List[str]) -> List[Dict]:
        """KGGen의 'Generate' 단계: 텍스트에서 지식 트리오 추출"""
        if not self.api_key:
            return []

        client = genai.Client(api_key=self.api_key)
        combined_text = "\n".join(text_list[:15]) # 상위 15개 뉴스만 샘플링
        
        prompt = f"""
        당신은 'Antigravity' 프로젝트의 시니어 데이터 엔지니어입니다.
        KGGen 기술을 사용하여 다음 EPL 뉴스에서 [주체 - 관계 - 객체] 형태의 지식 그래프를 추출하세요.
        축구 전술, 이적설, 부상, 감독 거취에 집중하세요.

        [뉴스 리스트]:
        {combined_text}

        [출력 예시 (JSON List)]:
        [
            {{"subject": "Manchester United", "relation": "interested_in", "object": "Michael Carrick", "type": "Manager_Link"}},
            {{"subject": "Marc Guehi", "relation": "full_agreement_with", "object": "Manchester City", "type": "Transfer"}}
        ]
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=prompt
            )
            # JSON 응답 정제
            raw_text = response.text
            print(f"DEBUG: Raw response: {raw_text[:200]}...")
            if '```json' in raw_text:
                raw_json = raw_text.split('```json')[1].split('```')[0].strip()
            elif '```' in raw_text:
                raw_json = raw_text.split('```')[1].strip()
            else:
                raw_json = raw_text.strip()
            
            return json.loads(raw_json)
        except Exception as e:
            print(f"KG Extraction Error: {e}")
            return []

    def update_knowledge_base(self):
        """KGGen의 'Aggregate' 단계: 새로운 지식을 기존 그래프에 통합"""
        # 1. 뉴스 데이터 로드
        news_path = "epl_project/data/latest_epl_data.json"
        if not os.path.exists(news_path):
            return
            
        with open(news_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            news_titles = [n['title'] for n in data.get('news', [])]

        # 2. 새로운 지식 추출
        new_triples = self.extract_graph_from_text(news_titles)
        
        # 3. 기존 데이터와 병합 (간단한 중복 제거)
        existing_graph = []
        if os.path.exists(self.graph_file):
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                existing_graph = json.load(f)
        
        # Merge logic (Simple)
        seen = { (t['subject'], t['relation'], t['object']) for t in existing_graph }
        for t in new_triples:
            if (t['subject'], t['relation'], t['object']) not in seen:
                existing_graph.append(t)
                
        # 4. 저장
        os.makedirs(os.path.dirname(self.graph_file), exist_ok=True)
        with open(self.graph_file, 'w', encoding='utf-8') as f:
            json.dump(existing_graph, f, ensure_ascii=False, indent=4)
        
        print(f"✅ 지식 그래프 업데이트 완료: {len(existing_graph)}개의 관계 저장됨.")

if __name__ == "__main__":
    engine = FootballKGEngine()
    engine.update_knowledge_base()
