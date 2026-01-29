import os
import polars as pl
import json
import subprocess
from datetime import datetime

# ==========================================
# ⚡ EPL Advanced Pipeline: Polars + MinerU + Grafana
# ==========================================

def run_mineru_extraction(pdf_path: str, output_dir: str):
    """
    MinerU(magic-pdf)를 사용하여 PDF에서 텍스트 및 구조화된 데이터를 추출합니다.
    """
    print(f"🚀 MinerU 실행 중: {pdf_path} 분석 시작...")
    try:
        # magic-pdf pdf --pdf <path> 형식으로 수정
        result = subprocess.run(
            ["magic-pdf", "pdf", "--pdf", pdf_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ MinerU 추출 완료!")
        else:
            print(f"⚠️ MinerU 경고: {result.stderr}")
    except Exception as e:
        print(f"❌ MinerU 실행 실패: {e}")

def process_with_polars(extracted_json_path: str, epl_data_path: str):
    """
    Polars를 사용하여 추출된 인사이트와 EPL 경기 데이터를 결합합니다.
    """
    print("📊 Polars 데이터 프로세싱 가동...")
    
    # 1. EPL 기본 데이터 로드 (Dummy or Real)
    # 여기서는 예시를 위해 간단한 데이터를 생성합니다.
    epl_df = pl.DataFrame({
        "team": ["Arsenal", "Man City", "Liverpool", "Aston Villa", "Spurs"],
        "points": [52, 53, 54, 46, 44],
        "goals_scored": [48, 54, 52, 49, 51],
        "tactical_discipline": [0.85, 0.90, 0.88, 0.75, 0.80]
    })

    # 2. MinerU에서 추출된 텍스트 인사이트 로드 (JSON 시뮬레이션)
    # 실제로는 추출된 JSON 파일을 파싱하여 특정 키워드(예: 전술 점수)를 뽑아냅니다.
    tactical_insights = {
        "Man City": {"innovation_score": 0.95, "scouting_report": "High press focus"},
        "Arsenal": {"innovation_score": 0.88, "scouting_report": "Positional play mastery"},
        "Liverpool": {"innovation_score": 0.92, "scouting_report": "Heavy metal football"}
    }
    
    insight_df = pl.DataFrame([
        {"team": team, "innovation_score": data["innovation_score"], "note": data["scouting_report"]}
        for team, data in tactical_insights.items()
    ])

    # 3. Polars Join (High Performance)
    final_df = epl_df.join(insight_df, on="team", how="left").fill_null(0.5)

    # 4. 파생 지표 계산
    final_df = final_df.with_columns([
        (pl.col("points") * pl.col("innovation_score")).alias("weighted_performance"),
        (pl.col("goals_scored") / 20).alias("attacking_index")
    ])

    # 5. 결과 저장 (Grafana가 읽기 쉬운 CSV/Parquet)
    output_path = "data/epl_final_insights.csv"
    final_df.write_csv(output_path)
    print(f"💾 최종 데이터 저장 시간: {datetime.now()}")
    print(f"📍 저장 위치: {output_path}")
    return final_df

def main():
    # 경로 설정
    current_dir = os.getcwd()
    sample_pdf = os.path.join(current_dir, "Kmong_Proposal_Phase1.pdf")
    output_base = os.path.join(current_dir, "output/mineru_results")
    
    # 1단계: MinerU 리서치 (PDF -> Insights)
    # (실제 대형 PDF가 없으면 이 단계는 시뮬레이션으로 작동할 수 있습니다)
    if os.path.exists(sample_pdf):
        run_mineru_extraction(sample_pdf, output_base)
    
    # 2단계: Polars 엔지니어링 (Processing)
    final_data = process_with_polars(output_base, "")

    # 3단계: Grafana 브리핑
    print("\n" + "="*50)
    print("📈 Grafana 시각화 가이드")
    print("="*50)
    print("1. Grafana 서버 실행: ./tools/grafana-12.3.1/bin/grafana server")
    print("2. http://localhost:3000 접속 (admin/admin)")
    print("3. 'CSV Data Source' 플러그인 설치 (또는 DuckDB 연동)")
    print(f"4. 데이터 소스로 다음 파일 지정: {os.path.join(current_dir, 'data/epl_final_insights.csv')}")
    print("5. 대시보드에서 'Weighted Performance'를 게이지 차트로 시각화하세요.")
    print("="*50)

if __name__ == "__main__":
    main()
