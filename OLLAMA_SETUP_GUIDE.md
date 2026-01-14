# 🤖 VS Code + Ollama(로컬 AI) 연동 가이드

시니어 분석가님, 설치하신 로컬 AI 모델들을 VS Code에서 사용하려면 **"Continue"** 라는 확장 프로그램을 설치하시는 것을 추천합니다. 
이것은 VS Code 안에서 작동하는 오픈소스 AI 어시스턴트입니다.

## 1. 확장 프로그램 설치
1. VS Code 좌측의 **Extensions(블럭 모양 아이콘)** 클릭
2. 검색창에 **`Continue`** 입력
3. **`Continue - Code Llama, GPT-4, and more`** 설치

## 2. 설정 파일 수정 (`config.json`)
설치 후, `~/.continue/config.json` 파일을 열고 아래 내용을 붙여넣으세요. 그러면 방금 설치한 모델들이 VS Code에 등록됩니다.

```json
{
  "models": [
    {
      "title": "Microsoft Phi-3.5 (가성비)",
      "provider": "ollama",
      "model": "phi3:medium"
    },
    {
      "title": "Google Gemma 2 (똑똑함)",
      "provider": "ollama",
      "model": "gemma2:9b"
    },
    {
      "title": "Qwen 2.5 Coder (코딩 전문)",
      "provider": "ollama",
      "model": "qwen2.5-coder:7b"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Qwen 2.5 Coder",
    "provider": "ollama",
    "model": "qwen2.5-coder:7b"
  }
}
```

## 3. 사용법
- **채팅하기**: `Cmd + L` 을 누르면 우측에 채팅창이 열립니다. 모델을 선택하고 질문하세요.
- **코드 자동완성**: 코드를 칠 때 자동으로 회색 텍스트가 뜹니다. `Tab`을 누르면 완성됩니다.
- **코드 수정**: 코드를 드래그하고 `Cmd + I`를 눌러 "이 코드 주석 달아줘" 같은 명령을 내리세요.

이제 인터넷 연결 없이 강력한 AI 짝꿍과 함께 코딩하세요! 🚀
