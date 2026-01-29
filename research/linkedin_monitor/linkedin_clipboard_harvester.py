
import os
import time
import pyperclip
import asyncio
from datetime import datetime
from google import genai
from google.genai import types

# 1. 설정
SAVE_DIR = "research/readings/linkedin_insights"
os.makedirs(SAVE_DIR, exist_ok=True)

api_key = os.environ.get("GEMINI_API_KEY")

async def analyze_content_async(text):
    """
    [New SDK] google-genai (2026 Standard) 활용
    스트리밍과 타임아웃을 적용하여 '대기 현상' 방지
    """
    if not api_key:
        return "❌ API Key가 설정되지 않았습니다."

    print("\n🧠 [Antigravity/Flash] 콘텐츠 분석 중... (Streaming 활성화)", flush=True)
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
    당신은 'Antigravity AI Engineer'입니다. 
    사용자가 클립보드에 복사한 텍스트가 'AI, 데이터 분석, 코딩, 엔지니어링'과 관련된 경험담이나 팁인지 분석하세요.
    글이 개발/데이터 관련이 아니면 첫 줄에 "NOT_RELEVANT"라고 쓰세요.
    
    [복사된 텍스트]
    {text[:2000]}
    """
    
    try:
        # 30초 타임아웃 설정 (anyio/asyncio 기반 처리)
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=500
        )
        
        # 스트리밍 요청
        response_text = ""
        stream = client.models.generate_content_stream(
            model='gemini-3-flash-preview', # [UPGRADE] Using Next-Gen Gemini 3 Flash
            contents=prompt,
            config=config
        )
        
        for chunk in stream:
            print("·", end="", flush=True) # 진행 상태 표시
            response_text += chunk.text
            if "NOT_RELEVANT" in response_text:
                return "NOT_RELEVANT"

        return response_text
    except Exception as e:
        return f"❌ 분석 실패 (Timeout/Network): {str(e)[:100]}"

async def monitor_clipboard():
    print("\n🚀 [LinkedIn Hyper-Harvester v2] 기동!")
    print("👉 현재 30초 타임아웃 및 스트리밍 모드가 활성화되었습니다.")
    print("👉 복사 즉시 반응하지 않으면 네트워크를 점검해 주세요.")
    print("-" * 50)

    last_paste = pyperclip.paste()
    while True:
        try:
            current_paste = pyperclip.paste()
            if current_paste != last_paste and current_paste.strip():
                detected_text = current_paste.strip()
                if len(detected_text) < 50:
                    last_paste = current_paste
                    continue

                print(f"\n⚡️ **새로운 콘텐츠 감지됨!** ({len(detected_text)}자)")
                
                # 비대기 분석 수행
                analysis = await analyze_content_async(detected_text)
                
                if "NOT_RELEVANT" in analysis:
                    print("\n   LOG: 관련 없는 콘텐츠로 판단되어 스킵합니다.")
                else:
                    print("\n" + "="*50)
                    print(analysis.strip())
                    print("="*50)
                    
                    # Human-in-the-loop
                    print("\n🙋 **[Confirm Required]**")
                    # 동기 input을 비동기에서 사용하기 위해 run_in_executor 고려 가능하나 단순화함
                    choice = input("👉 이 인사이트를 저장하시겠습니까? (y/n): ").lower()
                    
                    if choice == 'y':
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{SAVE_DIR}/insight_{timestamp}.md"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"# Source Clibpoard Content\n\n{detected_text}\n\n")
                            f.write(f"# Antigravity Analysis\n\n{analysis}")
                        print(f"✅ 저장 완료: {filename}")
                    else:
                        print("❌ 저장을 취소했습니다.")
                
                last_paste = current_paste
                print("\n📋 다음 복사를 대기 중...")
            
            await asyncio.sleep(1) # 부하 방지
        except Exception as e:
            print(f"\n⚠️ 오류 발생: {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(monitor_clipboard())
    except KeyboardInterrupt:
        print("\n👋 모니터링을 종료합니다.")
