"""
Gemini 기반 K-리그 자동 분석 시스템
GEMINI.md Protocol 준수 - 30년 차 시니어 분석가 페르소나
"""

import os
import pandas as pd
import google.generativeai as genai
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime


class GeminiKLeagueAnalyst:
    """Gemini API를 활용한 K-리그 전문 분석가"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            api_key: Gemini API 키 (없으면 환경변수에서 로드)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY가 필요합니다. "
                "환경변수 설정 또는 초기화 시 전달하세요."
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # GEMINI.md Protocol: 30년 차 시니어 분석가 페르소나
        self.system_prompt = """
당신은 30년 차 K-리그 전문 시니어 데이터 분석가입니다.

**핵심 원칙:**
1. [결론 - 근거 - 제언] 구조로 답변
2. 단순 통계 나열 금지, 반드시 맥락적 해석 포함
3. "왜 지금 이 데이터가 유의미한가?" 설명
4. 잠재적 결핍(Shadow KPI) 포착 및 제안
5. 한국어로 친절하지만 날카로운 인사이트 제공

**금지 사항:**
- "데이터가 부족합니다" 같은 회피성 답변
- 영어 전문 용어 남발 (한국어 쉬운 설명 우선)
- 단순 수치 요약 (반드시 인사이트 추가)
"""
    
    def analyze_team_performance(
        self, 
        df: pd.DataFrame, 
        team_name: str
    ) -> Dict[str, str]:
        """
        특정 팀의 성적 심층 분석
        
        Args:
            df: K-리그 데이터프레임
            team_name: 분석할 팀명
            
        Returns:
            분석 결과 딕셔너리 (conclusion, evidence, recommendation)
        """
        # 팀 데이터 필터링
        team_data = df[df['팀명'] == team_name].copy()
        
        if team_data.empty:
            return {
                "error": f"'{team_name}' 팀 데이터를 찾을 수 없습니다.",
                "available_teams": df['팀명'].unique().tolist()
            }
        
        # 기초 통계 계산
        stats = {
            "총 경기수": len(team_data),
            "승리": int(team_data['승'].sum()) if '승' in team_data.columns else 0,
            "무승부": int(team_data['무'].sum()) if '무' in team_data.columns else 0,
            "패배": int(team_data['패'].sum()) if '패' in team_data.columns else 0,
            "득점": int(team_data['득점'].sum()) if '득점' in team_data.columns else 0,
            "실점": int(team_data['실점'].sum()) if '실점' in team_data.columns else 0,
        }
        
        # 승률 계산
        if stats["총 경기수"] > 0:
            stats["승률"] = round(stats["승리"] / stats["총 경기수"] * 100, 2)
        else:
            stats["승률"] = 0.0
        
        # Gemini에게 분석 요청
        prompt = f"""
{self.system_prompt}

**분석 대상:** {team_name}
**기초 통계:**
{json.dumps(stats, ensure_ascii=False, indent=2)}

**최근 5경기 데이터:**
{team_data.tail(5).to_string()}

위 데이터를 바탕으로 다음 구조로 분석하세요:

## 🎯 결론 (Conclusion)
- 한 문장으로 핵심 진단

## 📊 근거 (Evidence)
- 통계적 근거 3가지 (구체적 수치 포함)
- 최근 흐름 분석 (상승/하락/정체)

## 💡 제언 (Recommendation)
- 전술적 개선 방향 2가지
- Shadow KPI 제안 (언급되지 않았지만 중요한 지표)

**Why Now?** 섹션도 반드시 포함하여 현재 시점의 맥락을 설명하세요.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return {
                "team": team_name,
                "analysis": response.text,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": f"Gemini API 호출 실패: {str(e)}",
                "stats": stats
            }
    
    def compare_teams(
        self, 
        df: pd.DataFrame, 
        team1: str, 
        team2: str
    ) -> Dict[str, str]:
        """
        두 팀 비교 분석 (라이벌 매치 등)
        
        Args:
            df: K-리그 데이터프레임
            team1: 첫 번째 팀명
            team2: 두 번째 팀명
            
        Returns:
            비교 분석 결과
        """
        # 각 팀 데이터 추출
        data1 = df[df['팀명'] == team1]
        data2 = df[df['팀명'] == team2]
        
        if data1.empty or data2.empty:
            return {"error": "한 팀 이상의 데이터가 없습니다."}
        
        # 비교 통계
        comparison = {
            team1: {
                "승률": round(data1['승'].sum() / len(data1) * 100, 2) if '승' in data1.columns else 0,
                "평균득점": round(data1['득점'].mean(), 2) if '득점' in data1.columns else 0,
                "평균실점": round(data1['실점'].mean(), 2) if '실점' in data1.columns else 0,
            },
            team2: {
                "승률": round(data2['승'].sum() / len(data2) * 100, 2) if '승' in data2.columns else 0,
                "평균득점": round(data2['득점'].mean(), 2) if '득점' in data2.columns else 0,
                "평균실점": round(data2['실점'].mean(), 2) if '실점' in data2.columns else 0,
            }
        }
        
        prompt = f"""
{self.system_prompt}

**라이벌 매치 분석:** {team1} vs {team2}

**비교 통계:**
{json.dumps(comparison, ensure_ascii=False, indent=2)}

다음 구조로 승부 예측 및 전술 분석을 제공하세요:

## ⚔️ 승부 예측
- 예상 승자 및 확률 (근거 포함)

## 🎯 전술적 우위
- {team1}의 강점 2가지
- {team2}의 강점 2가지

## 🔍 승부처 (Key Battle)
- 경기를 결정할 핵심 요소 3가지

## 💡 베팅 인사이트 (선택사항)
- 데이터 기반 베팅 제안 (책임 있는 도박 전제)
"""
        
        try:
            response = self.model.generate_content(prompt)
            return {
                "matchup": f"{team1} vs {team2}",
                "analysis": response.text,
                "comparison": comparison,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Gemini API 호출 실패: {str(e)}"}
    
    def generate_league_overview(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        전체 리그 현황 분석
        
        Args:
            df: K-리그 전체 데이터프레임
            
        Returns:
            리그 전체 분석 결과
        """
        # 팀별 집계
        team_stats = df.groupby('팀명').agg({
            '승': 'sum',
            '무': 'sum',
            '패': 'sum',
            '득점': 'sum',
            '실점': 'sum'
        }).reset_index()
        
        team_stats['승률'] = (team_stats['승'] / 
                             (team_stats['승'] + team_stats['무'] + team_stats['패']) * 100).round(2)
        team_stats = team_stats.sort_values('승률', ascending=False)
        
        prompt = f"""
{self.system_prompt}

**K-리그 전체 현황 분석**

**팀별 순위 (승률 기준):**
{team_stats.to_string()}

다음 구조로 리그 전체 트렌드를 분석하세요:

## 🏆 파워 랭킹 Top 3
- 1위~3위 팀의 강점 분석

## ⚠️ 위기의 팀 Bottom 3
- 하위 3팀의 문제점 및 개선 방향

## 📈 리그 트렌드
- 올 시즌 주요 특징 3가지
- 전년 대비 변화 (있다면)

## 🔮 시즌 전망
- 우승 후보 예측 (확률 포함)
- 강등권 예상 팀

**Why Now?** 현재 시점에서 주목해야 할 핵심 이슈를 반드시 언급하세요.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return {
                "analysis": response.text,
                "rankings": team_stats.to_dict('records'),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Gemini API 호출 실패: {str(e)}"}
    
    def save_report(
        self, 
        analysis: Dict[str, str], 
        output_dir: str = "../reports"
    ) -> Path:
        """
        분석 결과를 마크다운 파일로 저장
        
        Args:
            analysis: 분석 결과 딕셔너리
            output_dir: 저장 디렉토리
            
        Returns:
            저장된 파일 경로
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"k_league_analysis_{timestamp}.md"
        filepath = output_path / filename
        
        # 마크다운 생성
        content = f"""# K-리그 AI 분석 리포트

**생성 시간:** {analysis.get('timestamp', 'N/A')}
**분석 엔진:** Gemini 2.0 Flash (Experimental)

---

{analysis.get('analysis', '분석 내용 없음')}

---

## 📊 원본 데이터

```json
{json.dumps(analysis.get('stats', {}), ensure_ascii=False, indent=2)}
```

---

*Generated by Gemini K-League Analyst*
*GEMINI.md Protocol v1.9*
"""
        
        filepath.write_text(content, encoding='utf-8')
        return filepath


# 사용 예시
if __name__ == "__main__":
    # API 키 설정 (환경변수 또는 직접 입력)
    # export GEMINI_API_KEY="your-api-key-here"
    
    analyst = GeminiKLeagueAnalyst()
    
    # 샘플 데이터 로드 (실제 경로로 변경 필요)
    # df = pd.read_csv("../data/k_league_2024.csv")
    
    # 팀 분석
    # result = analyst.analyze_team_performance(df, "전북 현대")
    # print(result['analysis'])
    
    # 리포트 저장
    # filepath = analyst.save_report(result)
    # print(f"리포트 저장 완료: {filepath}")
    
    print("✅ Gemini K-League Analyst 모듈 로드 완료")
    print("📖 사용법: analyst = GeminiKLeagueAnalyst(api_key='your-key')")
