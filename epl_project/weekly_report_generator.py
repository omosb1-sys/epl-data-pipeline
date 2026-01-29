"""
EPL 주간 리포트 자동화 시스템
GEMINI.md Protocol 준수 - 30년 차 PM의 자동화 전략
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import os
import sys

# Gemini 분석 모듈 임포트
sys.path.append(str(Path(__file__).parent.parent / "src"))
try:
    from gemini_k_league_analyst import GeminiKLeagueAnalyst
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class EPLWeeklyReportGenerator:
    """EPL 주간 리포트 자동 생성 엔진"""
    
    def __init__(self, gemini_api_key: str = None):
        """
        초기화
        
        Args:
            gemini_api_key: Gemini API 키
        """
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        if GEMINI_AVAILABLE and self.api_key:
            self.analyst = GeminiKLeagueAnalyst(api_key=self.api_key)
        else:
            self.analyst = None
        
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.reports_dir = self.base_dir / "reports" / "weekly"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def load_latest_data(self) -> dict:
        """최신 EPL 데이터 로드"""
        try:
            data_path = self.data_dir / "latest_epl_data.json"
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return {}
    
    def load_clubs_data(self) -> list:
        """구단 정보 로드"""
        try:
            clubs_path = self.data_dir / "clubs_backup.json"
            if clubs_path.exists():
                with open(clubs_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"구단 데이터 로드 실패: {e}")
            return []
    
    def analyze_weekly_trends(self) -> dict:
        """주간 트렌드 분석"""
        clubs = self.load_clubs_data()
        
        # 전력 지수 기준 Top 3 / Bottom 3
        sorted_clubs = sorted(clubs, key=lambda x: x.get('power_index', 0), reverse=True)
        
        top_3 = sorted_clubs[:3]
        bottom_3 = sorted_clubs[-3:]
        
        # 승률 계산
        for club in clubs:
            wins = club.get('wins', 0)
            draws = club.get('draws', 0)
            losses = club.get('losses', 0)
            total = wins + draws + losses
            club['win_rate'] = (wins / total * 100) if total > 0 else 0
        
        # 승률 변화가 큰 팀 (가정: 최근 5경기 기준)
        trending_up = sorted(clubs, key=lambda x: x.get('win_rate', 0), reverse=True)[:3]
        
        return {
            "top_3": top_3,
            "bottom_3": bottom_3,
            "trending_up": trending_up,
            "total_teams": len(clubs)
        }
    
    def generate_gemini_commentary(self, trends: dict) -> str:
        """Gemini로 전문가급 논평 생성"""
        if not self.analyst:
            return "Gemini API 키가 필요합니다."
        
        # 프롬프트 생성
        prompt = f"""
당신은 30년 차 EPL 전문 분석가입니다.

다음 주간 데이터를 바탕으로 [결론-근거-제언] 구조로 리포트를 작성하세요:

## 📊 이번 주 데이터

**파워 랭킹 Top 3:**
{chr(10).join([f"- {t['team_name']} (전력 지수: {t.get('power_index', 0)})" for t in trends['top_3']])}

**위기의 팀 Bottom 3:**
{chr(10).join([f"- {t['team_name']} (전력 지수: {t.get('power_index', 0)})" for t in trends['bottom_3']])}

**상승세 팀:**
{chr(10).join([f"- {t['team_name']} (승률: {t.get('win_rate', 0):.1f}%)" for t in trends['trending_up']])}

---

다음 구조로 작성하세요:

## 🏆 이번 주 핵심 이슈

### 1. 파워 랭킹 분석
- Top 3 팀의 강점 분석 (각 2문장)

### 2. 위기의 팀 진단
- Bottom 3 팀의 문제점 및 개선 방향 (각 2문장)

### 3. 주목할 상승세
- 승률 상승 팀의 변화 요인 분석

### 4. 다음 주 전망
- 주목해야 할 경기 3개 예측

**Why Now?** 섹션도 반드시 포함하여 현재 시점의 맥락을 설명하세요.
"""
        
        try:
            response = self.analyst.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"논평 생성 실패: {str(e)}"
    
    def generate_report(self) -> Path:
        """주간 리포트 생성"""
        print("📊 주간 리포트 생성 중...")
        
        # 1. 데이터 분석
        trends = self.analyze_weekly_trends()
        
        # 2. Gemini 논평 생성
        commentary = self.generate_gemini_commentary(trends)
        
        # 3. 뉴스 데이터 로드
        latest_data = self.load_latest_data()
        news = latest_data.get('news', [])[:10]  # 최신 10개
        
        # 4. 리포트 생성
        timestamp = datetime.now()
        week_num = timestamp.isocalendar()[1]
        filename = f"EPL_Weekly_Report_W{week_num}_{timestamp.strftime('%Y%m%d')}.md"
        filepath = self.reports_dir / filename
        
        content = f"""# 📊 EPL 주간 리포트

**Week {week_num}, {timestamp.strftime('%Y년 %m월 %d일')}**  
**생성 시간:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**분석 엔진:** Gemini 2.0 Flash + EPL Data Engine

---

{commentary}

---

## 📰 이번 주 주요 뉴스

"""
        
        for i, n in enumerate(news, 1):
            if isinstance(n, dict):
                content += f"{i}. [{n.get('title', 'N/A')}]({n.get('url', '#')})\n"
            else:
                content += f"{i}. {n}\n"
        
        content += f"""

---

## 📈 팀별 상세 데이터

### 🏆 파워 랭킹 Top 3

"""
        
        for i, team in enumerate(trends['top_3'], 1):
            content += f"""
#### {i}. {team['team_name']}
- **전력 지수:** {team.get('power_index', 0)}/100
- **현재 감독:** {team.get('manager_name', 'N/A')}
- **시즌 전적:** {team.get('wins', 0)}승 {team.get('draws', 0)}무 {team.get('losses', 0)}패
- **주 포메이션:** {team.get('tactics_formation', 'N/A')}

"""
        
        content += """
### ⚠️ 위기의 팀 Bottom 3

"""
        
        for i, team in enumerate(trends['bottom_3'], 1):
            content += f"""
#### {i}. {team['team_name']}
- **전력 지수:** {team.get('power_index', 0)}/100
- **현재 감독:** {team.get('manager_name', 'N/A')}
- **시즌 전적:** {team.get('wins', 0)}승 {team.get('draws', 0)}무 {team.get('losses', 0)}패

"""
        
        content += f"""

---

## 🎯 다음 주 액션 아이템

1. **주목할 경기:** 파워 랭킹 1위 vs 상승세 팀
2. **위기 팀 모니터링:** Bottom 3 팀의 감독 교체 가능성 체크
3. **이적 시장:** 겨울 이적 시장 마감 임박 (주요 영입 예상)

---

*Generated by EPL Weekly Report Generator*  
*GEMINI.md Protocol v1.9*  
*Powered by Gemini 2.0 Flash*
"""
        
        # 5. 파일 저장
        filepath.write_text(content, encoding='utf-8')
        print(f"✅ 리포트 저장 완료: {filepath}")
        
        return filepath
    
    def generate_pdf(self, md_path: Path) -> Path:
        """마크다운을 PDF로 변환 (선택사항)"""
        # TODO: markdown-pdf 라이브러리 사용
        # 현재는 마크다운만 제공
        return md_path


# 사용 예시
if __name__ == "__main__":
    generator = EPLWeeklyReportGenerator()
    report_path = generator.generate_report()
    print(f"\n📄 리포트 위치: {report_path}")
    print("\n✅ 주간 리포트 자동화 완료!")
