import json
import time
import requests
import os
from pathlib import Path

# [Architect Mode] Side-by-Side Model Evaluator
# Target: LG EXAONE 3.5 vs Alibaba Qwen 2.5

class ModelBattleground:
    def __init__(self, golden_set_path: str):
        self.endpoint = "http://localhost:11434/api/generate"
        self.golden_set_path = Path(golden_set_path)
        with open(self.golden_set_path, "r", encoding="utf-8") as f:
            self.golden_set = json.load(f)

    def ask_ai(self, model: str, prompt: str):
        print(f"🤖 {model} 분석 중...")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.3 # 분석의 일관성을 위해 낮은 온도 설정
            }
        }
        try:
            start_time = time.time()
            response = requests.post(self.endpoint, json=payload, timeout=120)
            elapsed = time.time() - start_time
            if response.status_code == 200:
                return response.json()['response'], elapsed
        except Exception as e:
            return f"Error: {str(e)}", 0
        return "Failed", 0

    def run_showdown(self):
        # 라이벌 매치 로드 (Arsenal vs Tottenham)
        match = next((m for m in self.golden_set if "Arsenal" in m['home']), self.golden_set[0])
        
        system_prompt = f"""
당신은 30년 차 EPL 전술 분석 전문가입니다. 
다음 매치에 대해 전술 패러다임과 핵심 승부처(xG, 모멘텀 기반)를 분석하세요.
언어: 한국어 (Senior Analyst Tone 사용)

[매치 정보]
홈팀: {match['home']}
원정팀: {match['away']}
특이사항: {match['description']}
필수 키워드: {', '.join(match['required_keywords'])}
"""

        models = ["exaone3.5:7.8b", "qwen2.5:7b"]
        results = {}

        for model in models:
            res, elapsed = self.ask_ai(model, system_prompt)
            results[model] = {
                "output": res,
                "speed": f"{elapsed:.2f}s"
            }

        # 결과 저장
        report_path = "epl_project/reports/ai_showdown_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ 벤치마크 보고서 생성 완료: {report_path}")
        return results

if __name__ == "__main__":
    # 프로젝트 경로 보정
    PROJ_ROOT = Path("/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터")
    GOLDEN_SET = PROJ_ROOT / "epl_project/internal/golden_set.json"
    
    battle = ModelBattleground(GOLDEN_SET)
    showdown_data = battle.run_showdown()
    
    for model, data in showdown_data.items():
        print(f"\n{'='*50}")
        print(f"🏆 MODEL: {model} (소요시간: {data['speed']})")
        print(f"{'-'*50}")
        print(data['output'][:500] + "...") # 요약 출력
