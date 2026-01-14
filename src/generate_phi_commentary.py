import subprocess
import pandas as pd
import os
import sys

# 프로젝트 루트를 경로에 추가
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.k_league_timesfm_forecast import KLeagueForecaster

def generate_phi_commentary():
    print("🧠 AI 분석 엔진(Neural Network) 가동 중...")
    forecaster = KLeagueForecaster(data_path=os.path.join(BASE_DIR, 'data/raw/match_info.csv'))
    report = forecaster.run_league_analysis()
    
    if report is None:
        print("데이터 부족으로 분석을 진행할 수 없습니다.")
        return

    # 결과를 텍스트로 정리
    report_text = report.to_string(index=False)
    
    # Phi 3.5에게 보낼 프롬프트 구성
    prompt = f"""
전문의 시니어 데이터 분석가로서 다음 K-리그 딥러닝 분석 결과를 상세히 논평해줘.
분석 모델: MLP(Multi-Layer Perceptron) 신경망 시계열 예측
분석 대상: K-리그 구단별 차기 라운드 예상 득점력

[분석 결과 데이터]
{report_text}

[요청 사항]
1. 위 순위가 시사하는 바를 '시니어 분석가'의 어조로 설명해줘.
2. 득점력이 높게 예측된 팀과 낮게 예측된 팀의 전략적 차이를 분석해줘.
3. K-리그 팬들에게 전하는 이번 라운드 관전 포인트를 3가지로 요약해줘.
4. 모든 답변은 한국어로 작성해줘.
"""

    print("🤖 Phi 3.5에게 논평 요청 중 (Ollama)...")
    
    try:
        # Ollama 실행 (phi3.5:latest 사용)
        result = subprocess.run(
            ['/usr/local/bin/ollama', 'run', 'phi3.5', prompt],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            commentary = result.stdout
            
            # 결과 저장
            output_path = os.path.join(BASE_DIR, "research/PHI3.5_KLEAGUE_COMMENTARY.md")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# 🤖 Phi 3.5의 K-리그 딥러닝 분석 논평\n\n")
                f.write(commentary)
            
            print(f"✅ 논평이 생성되었습니다: {output_path}")
            print("\n" + "="*50)
            print(commentary)
            print("="*50)
        else:
            print(f"❌ Ollama 실행 오류: {result.stderr}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    generate_phi_commentary()
