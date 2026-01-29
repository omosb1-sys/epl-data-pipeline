"""
Antigravity 통합: 자율 유저 테스트 시스템
Active Personas + Browser Automation = 24시간 자동 UX 검증
"""

from synthetic_user_research import ActivePersona
from pathlib import Path
import time
from datetime import datetime
import json


class AntigravityUserTester:
    """Antigravity 기반 자율 사용자 테스트 시스템"""
    
    def __init__(self, app_url: str):
        """
        초기화
        
        Args:
            app_url: 테스트할 앱 URL (예: "https://epl-data-2026.streamlit.app/")
        """
        self.app_url = app_url
        self.test_results = []
        self.screenshots_dir = Path("epl_project/test_screenshots")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    def run_persona_test(self, persona: ActivePersona, test_scenario: dict) -> dict:
        """
        페르소나 기반 사용자 테스트 실행
        
        Args:
            persona: ActivePersona 객체
            test_scenario: 테스트 시나리오
                {
                    "name": "회원가입 플로우",
                    "steps": [
                        {"action": "navigate", "target": "/signup"},
                        {"action": "screenshot", "name": "signup_page"},
                        {"action": "click", "target": "가입하기 버튼"},
                        ...
                    ]
                }
                
        Returns:
            테스트 결과
        """
        print(f"\n🧪 테스트 시작: {test_scenario['name']}")
        print(f"👤 페르소나: {persona.profile['name']}")
        
        results = {
            "persona": persona.profile['name'],
            "scenario": test_scenario['name'],
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "overall_feedback": ""
        }
        
        # 각 단계 실행
        for i, step in enumerate(test_scenario['steps']):
            print(f"\n  Step {i+1}: {step['action']}")
            
            if step['action'] == "screenshot":
                # 스크린샷 촬영 (실제로는 Antigravity가 수행)
                screenshot_path = self.screenshots_dir / f"{step['name']}.png"
                
                # 페르소나 반응 수집
                reaction = persona.react_to_ui(
                    str(screenshot_path),
                    context=f"{test_scenario['name']} - Step {i+1}"
                )
                
                results['steps'].append({
                    "step": i+1,
                    "action": step['action'],
                    "screenshot": str(screenshot_path),
                    "persona_reaction": reaction
                })
                
                print(f"    반응: {reaction.get('reaction', 'N/A')}")
                print(f"    다음 행동: {reaction.get('next_action', 'N/A')}")
                
                # 불편한 점 발견 시 즉시 기록
                if reaction.get('pain_points'):
                    print(f"    ⚠️ 불편한 점: {', '.join(reaction['pain_points'])}")
            
            elif step['action'] == "click":
                # 클릭 시뮬레이션 (실제로는 Antigravity가 수행)
                print(f"    클릭: {step['target']}")
                results['steps'].append({
                    "step": i+1,
                    "action": "click",
                    "target": step['target']
                })
            
            elif step['action'] == "input":
                # 입력 시뮬레이션
                print(f"    입력: {step['target']} = {step['value']}")
                results['steps'].append({
                    "step": i+1,
                    "action": "input",
                    "target": step['target'],
                    "value": step['value']
                })
        
        # 전체 피드백 수집
        overall_question = f"""
{test_scenario['name']} 테스트를 완료했습니다.
전체적인 경험을 평가해주세요:

1. 만족도 (1~10점):
2. 가장 좋았던 점:
3. 가장 불편했던 점:
4. 개선이 시급한 부분:
"""
        
        overall_feedback = persona.interview(overall_question)
        results['overall_feedback'] = overall_feedback
        
        print(f"\n✅ 테스트 완료")
        print(f"전체 피드백:\n{overall_feedback}")
        
        self.test_results.append(results)
        return results
    
    def run_multi_persona_test(self, personas: list, test_scenario: dict) -> list:
        """
        여러 페르소나로 동시 테스트
        
        Args:
            personas: ActivePersona 리스트
            test_scenario: 테스트 시나리오
            
        Returns:
            모든 페르소나의 테스트 결과
        """
        print(f"\n🚀 다중 페르소나 테스트 시작")
        print(f"페르소나 수: {len(personas)}명")
        
        all_results = []
        
        for persona in personas:
            result = self.run_persona_test(persona, test_scenario)
            all_results.append(result)
            time.sleep(1)  # API 레이트 리밋 방지
        
        # 통합 리포트 생성
        self.generate_summary_report(all_results, test_scenario['name'])
        
        return all_results
    
    def generate_summary_report(self, results: list, scenario_name: str):
        """
        통합 리포트 생성
        
        Args:
            results: 모든 테스트 결과
            scenario_name: 시나리오 이름
        """
        report_dir = Path("epl_project/reports/user_testing")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"user_test_{timestamp}.md"
        
        # 불편한 점 집계
        all_pain_points = []
        for result in results:
            for step in result['steps']:
                if 'persona_reaction' in step:
                    pain_points = step['persona_reaction'].get('pain_points', [])
                    all_pain_points.extend(pain_points)
        
        # 빈도 계산
        from collections import Counter
        pain_point_counts = Counter(all_pain_points)
        
        # 리포트 작성
        content = f"""# 🧪 사용자 테스트 리포트

**시나리오:** {scenario_name}
**테스트 일시:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**참여 페르소나:** {len(results)}명

---

## 📊 주요 발견사항

### 🔴 가장 많이 언급된 불편한 점 Top 5

"""
        
        for i, (pain_point, count) in enumerate(pain_point_counts.most_common(5), 1):
            content += f"{i}. **{pain_point}** ({count}명 언급)\n"
        
        content += "\n---\n\n## 👥 페르소나별 상세 피드백\n\n"
        
        for result in results:
            content += f"""
### {result['persona']}

**전체 피드백:**
{result['overall_feedback']}

**단계별 반응:**
"""
            for step in result['steps']:
                if 'persona_reaction' in step:
                    reaction = step['persona_reaction']
                    content += f"""
- **Step {step['step']}**
  - 첫인상: {reaction.get('reaction', 'N/A')}
  - 다음 행동: {reaction.get('next_action', 'N/A')}
  - 불편한 점: {', '.join(reaction.get('pain_points', []))}
"""
        
        content += """

---

## 🎯 개선 권장사항

"""
        
        # 모든 제안 수집
        all_suggestions = []
        for result in results:
            for step in result['steps']:
                if 'persona_reaction' in step:
                    suggestions = step['persona_reaction'].get('suggestions', [])
                    all_suggestions.extend(suggestions)
        
        # 중복 제거
        unique_suggestions = list(set(all_suggestions))
        
        for i, suggestion in enumerate(unique_suggestions[:10], 1):
            content += f"{i}. {suggestion}\n"
        
        content += """

---

*Generated by Antigravity User Testing System*  
*Active Personas × Gemini 2.0 Flash*
"""
        
        # 파일 저장
        report_path.write_text(content, encoding='utf-8')
        print(f"\n📄 리포트 저장: {report_path}")
        
        # JSON도 저장
        json_path = report_dir / f"user_test_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"📄 JSON 저장: {json_path}")


# 사용 예시
if __name__ == "__main__":
    # 페르소나 생성
    persona_profiles = [
        {
            "name": "김철수 (중급 사용자)",
            "age": 35,
            "occupation": "회사원",
            "football_team": "맨체스터 유나이티드",
            "tech_savvy": "중급",
            "pain_points": ["로딩 속도", "복잡한 UI"],
            "goals": ["빠른 경기 결과 확인", "승부 예측"]
        },
        {
            "name": "이영희 (고급 사용자)",
            "age": 28,
            "occupation": "마케터",
            "football_team": "리버풀",
            "tech_savvy": "고급",
            "pain_points": ["데이터 부족", "분석 깊이"],
            "goals": ["심층 전술 분석", "선수 통계"]
        },
        {
            "name": "박민수 (초급 사용자)",
            "age": 42,
            "occupation": "자영업",
            "football_team": "토트넘",
            "tech_savvy": "초급",
            "pain_points": ["어려운 용어", "복잡한 메뉴"],
            "goals": ["간단한 경기 일정 확인"]
        }
    ]
    
    personas = [ActivePersona(profile) for profile in persona_profiles]
    
    # 테스트 시나리오 정의
    test_scenario = {
        "name": "EPL 앱 첫 방문 경험",
        "steps": [
            {"action": "navigate", "target": "https://epl-data-2026.streamlit.app/"},
            {"action": "screenshot", "name": "home_page"},
            {"action": "click", "target": "팀 선택 드롭다운"},
            {"action": "screenshot", "name": "team_selector"},
            {"action": "click", "target": "맨체스터 유나이티드"},
            {"action": "screenshot", "name": "team_dashboard"},
            {"action": "click", "target": "AI 승부 예측 메뉴"},
            {"action": "screenshot", "name": "prediction_page"}
        ]
    }
    
    # 테스트 실행
    tester = AntigravityUserTester("https://epl-data-2026.streamlit.app/")
    results = tester.run_multi_persona_test(personas, test_scenario)
    
    print("\n✅ 모든 테스트 완료!")
    print(f"총 {len(results)}명의 페르소나 테스트 완료")
