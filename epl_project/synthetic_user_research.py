# 🎯 Active Personas for Synthetic User Research - Antigravity 적용

# "Gemini 두뇌 + Antigravity 몸 = 24시간 자율 유저 리서치"
# 기반: Active Personas 논문 x GEMINI.md Protocol v3.0
# 날짜: 2026-01-18 20:20 KST

import google.generativeai as genai
import os
import json

class ActivePersona:
    """Gemini 기반 활성 페르소나"""
    
    def __init__(self, persona_profile: dict, api_key: str = None):
        self.profile = persona_profile
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            self.model = None
        
        self.system_prompt = self._generate_system_prompt()
    
    def _generate_system_prompt(self) -> str:
        return f"""
당신은 다음 프로필을 가진 실제 사용자입니다:

**이름:** {self.profile['name']}
**나이:** {self.profile['age']}세
**직업:** {self.profile['occupation']}
**좋아하는 팀:** {self.profile['football_team']}
**기술 수준:** {self.profile['tech_savvy']}
**불편한 점:** {', '.join(self.profile['pain_points'])}
**목표:** {', '.join(self.profile['goals'])}

당신은 EPL 분석 앱을 사용하는 실제 사용자처럼 행동하고 반응합니다.
UI를 보고, 클릭하고, 불편한 점을 솔직하게 말합니다.
당신의 성격과 목표에 맞게 자연스럽게 대화하세요.
"""
    
    def react_to_ui(self, screenshot_path: str, context: str) -> dict:
        if not self.model: return {"error": "API Key missing"}
        
        uploaded_file = genai.upload_file(screenshot_path)
        
        prompt = f"""
{self.system_prompt}

**현재 상황:** {context}

위 스크린샷을 보고 다음 질문에 답하세요:

1. **첫인상**: 이 화면을 보고 어떤 느낌이 드나요?
2. **다음 행동**: 무엇을 클릭하거나 입력하고 싶나요?
3. **불편한 점**: 무엇이 불편하거나 혼란스러운가요?
4. **개선 제안**: 어떻게 하면 더 좋을까요?

JSON 형식으로 답변하세요:
{{
    "reaction": "첫인상",
    "next_action": "클릭/스크롤/입력/이탈",
    "target": "클릭할 요소",
    "pain_points": [],
    "suggestions": []
}}
"""
        response = self.model.generate_content([uploaded_file, prompt])
        try:
            return json.loads(response.text)
        except:
            return {"reaction": response.text}
    
    def interview(self, question: str) -> str:
        if not self.model: return "API Key missing"
        prompt = f"{self.system_prompt}\n\n**질문:** {question}"
        response = self.model.generate_content(prompt)
        return response.text

if __name__ == "__main__":
    persona_profiles = [
        {
            "name": "김철수", "age": 35, "occupation": "회사원", "football_team": "맨체스터 유나이티드",
            "tech_savvy": "중급", "pain_points": ["로딩 속도", "복잡한 UI"], "goals": ["빠른 경기 결과 확인", "승부 예측"]
        }
    ]
    personas = [ActivePersona(profile) for profile in persona_profiles]
    for persona in personas:
        print(f"\n=== {persona.profile['name']} 인터뷰 ===")
        # API 키가 있을 때만 실행
        if persona.model:
            print(persona.interview("EPL 앱에서 가장 중요한 기능은 무엇인가요?"))
        else:
            print("Skip: API Key not found")
