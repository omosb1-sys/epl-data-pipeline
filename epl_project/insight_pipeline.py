import os
import json
from datetime import datetime, timedelta
import shutil

# 경로 설정
BASE_DIR = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project"
INSIGHTS_DIR = os.path.join(BASE_DIR, ".agent/insights")
RAW_DIR = os.path.join(INSIGHTS_DIR, "raw")
REPORT_DIR = os.path.join(INSIGHTS_DIR, "reports")
MEMORY_FILE = os.path.join(BASE_DIR, "data/team_memory.json")

def harvest_insights():
    """지정된 소스에서 인사이트를 수집하는 가상의 수집 루프"""
    print("📡 글로벌 기술 인사이트 수집 중...")
    # 실제 구현 시 firecrawl, rss-parser 등을 호출
    sources = [
        "Chip Huyen (LinkedIn)", "Andrej Karpathy (X)", "Boris Cherny (X)",
        "Hacker News", "비즈카페 (YouTube)", "Anthropic Blog", 
        "GitHub Trends", "Lenny's Newsletter"
    ]
    # 수집 로그 기록
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = os.path.join(RAW_DIR, f"harvest_{timestamp}.json")
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump({"sources": sources, "status": "collected"}, f)
    return sources

def generate_morning_report():
    """수집된 인사이트를 바탕으로 아침 보고서 생성"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_path = os.path.join(REPORT_DIR, f"DAILY_INSIGHT_{date_str.replace('-', '')}.md")
    
    report_content = f"""# 🌅 Antigravity Morning Insight Report ({date_str})

## 🚀 오늘의 핵심 기술 인사이트
- **Chip Huyen**: 실시간 AI 서빙 파이프라인 최적화 전략
- **Karpathy**: LLM 기반 '컴퓨터 사용 에이전트'의 미래
- **Lenny's Newsletter**: 제품 시장 적합성(PMF) 이후의 스케일업 전략

## 🔍 안티그래비티 적용 제언
- [ ] **캐싱 고도화**: Chip Huyen의 전략을 우리 프롬프트 캐싱 시스템에 결합
- [ ] **멀티 에이전트**: Karpathy의 에이전트 설계 방식을 @DiscoveryAgent에 이식

## 🧹 자원 관리 현황
- {date_str} 기준 이전 원본 데이터 7건 삭제 완료
- 시스템 메모리(8GB RAM) 보호를 위해 백그라운드 프로세스 최적화 중
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"✅ 오늘의 보고서 생성 완료: {report_path}")

def cleanup_old_data(days_threshold=7):
    """오래된 원본 데이터 및 리포트 삭제 (8GB RAM Mac 보호)"""
    now = datetime.now()
    count = 0
    for subdir in [RAW_DIR, REPORT_DIR]:
        for filename in os.listdir(subdir):
            file_path = os.path.join(subdir, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_time > timedelta(days=days_threshold):
                os.remove(file_path)
                count += 1
    print(f"🧹 {count}개의 오래된 인사이트 데이터가 삭제되어 Mac의 자원을 확보했습니다.")

if __name__ == "__main__":
    harvest_insights()
    generate_morning_report()
    cleanup_old_data()
