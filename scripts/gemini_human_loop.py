
import os
import google.generativeai as genai
from typing import List, Dict, Any

# 제미나이 API 키 로드 (환경 변수에서 가져오거나 직접 입력)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # 사용자 편의를 위해 키가 없으면 입력을 받도록 안내 (실제 앱에서는 .env 권장)
    print("⚠️  GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("👉 구글 AI Studio에서 발급받은 키를 입력해주세요 (입력 내용은 숨겨지지 않습니다)")
    api_key = input("🔑 GEMINI_API_KEY 입력: ").strip()

# Gemini 모델 설정
genai.configure(api_key=api_key)

# 1. ask_user 도구 정의
tools_config = [
    {
        "function_declarations": [
            {
                "name": "ask_user",
                "description": "사용자에게 추가 정보나 확인을 요청할 때 사용합니다. 모델이 스스로 결정하기 어려운 모호한 상황에서 필수적으로 호출해야 합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "사용자에게 물어볼 구체적인 질문 내용"
                        }
                    },
                    "required": ["question"]
                }
            }
            # 여기에 다른 도구들(예: read_file 등)을 추가할 수 있습니다.
        ]
    }
]

# 시스템 프롬프트 설정 (Human-in-the-loop 강제화)
system_instruction = """
당신은 'Antigravity' 에이전트입니다. 
사용자의 요청을 수행하되, 모호한 부분이 있거나 중요한 결정(파일 삭제, 대규모 변경 등)이 필요할 때는 
반드시 'ask_user' 도구를 사용하여 확인을 받아야 합니다.
절대 추측하여 행동하지 말고, 사용자에게 명확한 가이드를 요청하세요.
"""

model = genai.GenerativeModel('gemini-1.5-pro', tools=tools_config, system_instruction=system_instruction)
chat = model.start_chat(history=[])

# 2. 실행 루프 구현 (Interactive Loop with Function Calling)
def run_interactive_agent():
    print("🤖 Antigravity Human-in-the-loop Agent 시작 (종료하려면 'exit' 입력)")
    print("-" * 50)
    
    while True:
        user_input = input("\n👤 사용자: ")
        if user_input.lower() in ['exit', 'quit', '종료']:
            print("👋 에이전트를 종료합니다.")
            break
            
        try:
            print("⏳ 지니가 생각 중...", end="\r", flush=True)  # 상태 표시 추가
            response = chat.send_message(user_input)
            print(" " * 20, end="\r", flush=True)  # 상태 표시 지우기
            
            # 모델 응답 처리 루프 (도구 호출이 없을 때까지 반복)
            while True:
                part = response.candidates[0].content.parts[0]
                
                # Function Call 확인
                if part.function_call:
                    call = part.function_call
                    
                    if call.name == "ask_user":
                        # 1. 모델의 질문 출력
                        question = call.args["question"]
                        print(f"\n❔ [Gemini 질문]: {question}")
                        
                        # 2. 사용자 답변 받기
                        user_answer = input("=> 🗣️ 답변: ")
                        
                        # 3. 답변을 모델에 전달 (Function Response)
                        response = chat.send_message(
                            genai.types.Content(
                                parts=[genai.types.Part.from_function_response(
                                    name="ask_user",
                                    response={"answer": user_answer}
                                )]
                            )
                        )
                    else:
                        # 다른 함수 호출 처리 (예시: 아직 구현되지 않은 함수)
                        print(f"\n⚙️ [도구 실행 요청]: {call.name}")
                        # 실제로는 여기서 해당 함수 로직을 실행하고 결과를 돌려줘야 함
                        # 임시로 성공 메시지 반환
                        response = chat.send_message(
                            genai.types.Content(
                                parts=[genai.types.Part.from_function_response(
                                    name=call.name,
                                    response={"result": "Success (Mock)"}
                                )]
                            )
                        )
                else:
                    # 도구 호출이 아닌 일반 텍스트 응답인 경우 출력하고 루프 종료
                    print(f"\n🤖 Antigravity: {response.text}")
                    break
                    
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    run_interactive_agent()
