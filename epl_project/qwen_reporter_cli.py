import json
import subprocess
from datetime import datetime

def generate_report_cli():
    print("🧠 분석 에이전트 군단이 토론 중... (Qwen 2.5 1.5B CLI 기반)")
    
    try:
        with open('data/latest_epl_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 뉴스 5개로 제한하여 컨텍스트 축소
    news_list = data.get('news', [])[:5]
    news_text = "\n".join([f"- {n['title']}" for n in news_list])
    
    prompt = f"""당신은 30년 차 시니어 EPL 데이터 분석가입니다. 
뉴스: {news_text}
위 뉴스를 기반으로 [결론-근거-제언] 리포트를 한글로 짧고 굵게 작성해 주세요. 맨시티의 게히 영입과 한국 선수 소식을 위주로 다뤄주세요."""

    try:
        # subprocess를 사용하여 직접 ollama 실행 (스트리밍)
        process = subprocess.Popen(
            ['ollama', 'run', 'qwen2.5:1.5b', prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        print("\n" + "="*50)
        print("📊 EPL WEEKLY INTELLIGENCE REPORT (STREAMING)")
        print("="*50)
        print(f"⏱️ 분석 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report = ""
        while True:
            output = process.stdout.read(1)
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output, end='', flush=True)
                report += output
        
        print("\n" + "="*50)
        
        with open('reports/qwen_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(f"# 📊 EPL Weekly Intelligence Report\n\n{report}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    generate_report_cli()
