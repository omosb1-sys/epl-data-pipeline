"""
Active Personas 시뮬레이션 테스트
실제 브라우저 없이 페르소나 피드백 수집
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from synthetic_user_research import ActivePersona
import json
from datetime import datetime


def simulate_ui_screenshot_analysis():
    """UI 스크린샷 분석 시뮬레이션"""
    
    # 페르소나 프로필 정의
    persona_profiles = [
        {
            "name": "김철수",
            "age": 35,
            "occupation": "회사원",
            "football_team": "맨체스터 유나이티드",
            "tech_savvy": "중급",
            "pain_points": ["로딩 속도", "복잡한 UI"],
            "goals": ["빠른 경기 결과 확인", "승부 예측"]
        },
        {
            "name": "이영희",
            "age": 28,
            "occupation": "마케터",
            "football_team": "리버풀",
            "tech_savvy": "고급",
            "pain_points": ["데이터 부족", "분석 깊이"],
            "goals": ["심층 전술 분석", "선수 통계"]
        },
        {
            "name": "박민수",
            "age": 42,
            "occupation": "자영업",
            "football_team": "토트넘",
            "tech_savvy": "초급",
            "pain_points": ["어려운 용어", "복잡한 메뉴"],
            "goals": ["간단한 경기 일정 확인"]
        }
    ]
    
    print("=" * 60)
    print("🧪 EPL 앱 사용자 테스트 시뮬레이션")
    print("=" * 60)
    
    # 테스트 시나리오
    scenarios = [
        {
            "name": "홈페이지 첫인상",
            "context": "EPL 앱 홈페이지에 처음 접속했습니다. 화면에는 팀 선택 드롭다운, AI 승부 예측 메뉴, 최신 뉴스 섹션이 보입니다."
        },
        {
            "name": "팀 대시보드",
            "context": "맨체스터 유나이티드를 선택했습니다. 화면에는 팀 전력 지수(91/100), 현재 감독(판 니스텔루이), ADX 모멘텀 차트가 표시됩니다."
        },
        {
            "name": "AI 승부 예측",
            "context": "AI 승부 예측 메뉴에 들어왔습니다. 맨유 vs 리버풀 경기의 예측 확률이 표시되지만, 근거나 신뢰도 점수는 보이지 않습니다."
        }
    ]
    
    all_feedback = []
    
    # 각 페르소나별 테스트
    for profile in persona_profiles:
        print(f"\n{'='*60}")
        print(f"👤 페르소나: {profile['name']} ({profile['tech_savvy']} 사용자)")
        print(f"{'='*60}")
        
        persona = ActivePersona(profile)
        persona_feedback = {
            "persona": profile['name'],
            "tech_level": profile['tech_savvy'],
            "scenarios": []
        }
        
        for scenario in scenarios:
            print(f"\n📍 시나리오: {scenario['name']}")
            print(f"   상황: {scenario['context']}")
            
            # 페르소나 인터뷰
            question = f"""
다음 상황에서 당신의 반응을 알려주세요:

**상황:** {scenario['context']}

다음 질문에 답해주세요:
1. 첫인상은 어떤가요? (1~10점)
2. 무엇이 가장 눈에 띄나요?
3. 불편하거나 혼란스러운 점은?
4. 다음에 무엇을 하고 싶나요?
5. 개선 제안이 있다면?

간결하게 답변해주세요.
"""
            
            try:
                response = persona.interview(question)
                print(f"\n   💬 반응:\n{response}")
                
                persona_feedback['scenarios'].append({
                    "scenario": scenario['name'],
                    "response": response
                })
            except Exception as e:
                print(f"\n   ❌ 오류: {e}")
                persona_feedback['scenarios'].append({
                    "scenario": scenario['name'],
                    "response": f"오류 발생: {e}"
                })
        
        all_feedback.append(persona_feedback)
    
    # 결과 저장
    output_dir = Path("epl_project/reports/user_testing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"simulation_test_{timestamp}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_feedback, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 테스트 완료!")
    print(f"📄 결과 저장: {output_path}")
    print(f"{'='*60}")
    
    return all_feedback


if __name__ == "__main__":
    feedback = simulate_ui_screenshot_analysis()
    
    print(f"\n📊 총 {len(feedback)}명의 페르소나 테스트 완료")
