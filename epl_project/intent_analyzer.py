"""
EPL User Intent Analyzer (Real Purchase Intent Prototype)
Lior Alex의 'Activity based Intent' 원칙을 EPL 분석 앱에 적용한 엔진.
단순 방문자가 아닌, 특정 데이터(전술, 예측, 심화 스태츠)를 깊게 파고드는 유저를 식별하여 
개인화된 인사이트나 프리미엄 리포트를 제안합니다.
"""

import json
import os
from datetime import datetime
from pathlib import Path

class IntentAnalyzer:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.audit_file = self.base_dir / "data" / "prediction_audit.jsonl"
        self.intent_report_dir = self.base_dir / "reports" / "intent"
        self.intent_report_dir.mkdir(parents=True, exist_ok=True)

    def analyze_activity_signal(self):
        """
        사용자의 활동 로그를 분석하여 'High Intent' 시그널을 추출합니다.
        시뮬레이션: prediction_audit.jsonl의 빈도와 복잡도를 분석.
        """
        print("🔍 [Intent Signal Detection] 사용자 활동 패턴 분석 중...")
        
        if not self.audit_file.exists():
            return "No audit data found."

        high_intent_count = 0
        complex_query_count = 0
        
        with open(self.audit_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                high_intent_count += 1
                
                # 'vars'(injured, rest 등)를 포함한 정밀 예측 시도는 'High Intent'로 간주
                if 'vars' in record.get('data', {}):
                    complex_query_count += 1

        # Lior Alex의 원칙 적용: 특정 행동(Activity)의 임계값(Threshold) 설정
        is_power_user = (high_intent_count > 5) and (complex_query_count > 2)
        
        return {
            "total_interactions": high_intent_count,
            "complex_analyses": complex_query_count,
            "is_power_user": is_power_user,
            "timestamp": datetime.now().isoformat()
        }

    def generate_personalized_offer(self, signal: dict):
        """
        사용자 의도에 기반한 개인화된 제안(Call to Action) 생성
        """
        if signal["is_power_user"]:
            offer = """
### 🚀 Premium Insight for YOU
당신은 현재 단순한 팬을 넘어 **'Deep Data Analyst'**의 행보를 보이고 있습니다.
특히 부상자 및 휴식일 변수를 활용한 정밀 예측 기능을 3회 이상 사용하셨네요.

**진짜 구매 의도 감지:**
당신 같은 전문가를 위해 **'EPL 전술 X-Ray 리포트'** 정기 구독권을 30% 할인된 가격에 제안합니다.
단순 결과 예측을 넘어, 감독의 하프타임 전술 변화까지 실시간으로 알림 받으세요!
            """
        else:
            offer = """
### ⚽ Enjoy the Game!
다양한 팀의 예측 결과를 확인하고 계시군요.
더 정확한 승부 예측을 위해 **'부상자 명단 반영 기능'**을 한번 사용해보시는 건 어떨까요?
            """
        
        return offer

    def run_protocol(self):
        signal = self.analyze_activity_signal()
        offer = self.generate_personalized_offer(signal)
        
        report_path = self.intent_report_dir / f"intent_report_{datetime.now().strftime('%Y%m%d')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Intent Analysis Report\n\nGenerated: {signal['timestamp']}\n\n{offer}")
            
        print(f"✅ Intent Protocol 가동 완료: {report_path}")
        return offer

if __name__ == "__main__":
    analyzer = IntentAnalyzer()
    print(analyzer.run_protocol())
