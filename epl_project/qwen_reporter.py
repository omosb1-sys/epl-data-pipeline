import json
import ollama
from datetime import datetime

def generate_report():
    print("🧠 분석 에이전트 군단이 토론 중... (Qwen 2.5 1.5B 기반)")
    
    # 데이터 로드
    try:
        with open('data/latest_epl_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    news_text = "\n".join([f"- {n['title']} (Source: {n['source']})" for n in data.get('news', [])])
    transfer_text = "\n".join([f"- {t['player']}: {t['from']} -> {t['to']} ({t['type']})" for t in data.get('transfers', [])])
    
    prompt = f"""
당신은 30년 차 시니어 프리미어리그 데이터 분석가입니다. 
다음은 최근 수집된 EPL 관련 뉴스 및 이적 정보입니다.

[최신 뉴스]
{news_text}

[이적 현황]
{transfer_text}

이 데이터를 기반으로 'EPL 위클리 인텔리전스 리포트'를 작성해 주세요. 
보고서는 반드시 [결론 - 근거 - 제언] 구조를 갖추어야 하며, 다음 내용을 포함해야 합니다:

1. 맨시티의 최근 이적 시장 행보(마크 게히, 세메뉴)에 대한 전술적 평가.
2. 코리안 리거(황희찬, 손흥민) 관련 소식에 대한 분석.
3. 전반적인 리그 흐름에 대한 날카로운 지적.

말투는 친절하지만 시니어다운 권위와 통찰력이 느껴져야 합니다. 한글로 작성해 주세요.
"""

    try:
        response = ollama.chat(model='qwen2.5:1.5b', messages=[
            {'role': 'user', 'content': prompt}
        ])
        
        report = response['message']['content']
        
        print("\n" + "="*50)
        print("📊 EPL WEEKLY INTELLIGENCE REPORT")
        print("="*50)
        print(f"⏱️ 분석 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + report)
        print("\n" + "="*50)
        
        # 파일로 저장
        with open('reports/qwen_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(f"# 📊 EPL Weekly Intelligence Report\n\n")
            f.write(f"*⏱️ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(report)
            
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    generate_report()
