# Stitch MCP Setup Guide

The configuration for Stitch MCP has been added to `~/.gemini/antigravity/mcp_config.json`. Since `gcloud` was not found on your system, please follow these steps from the beginning:

## 0. Install Google Cloud SDK (macOS)
If you haven't installed the Google Cloud SDK yet, run this in your terminal to download and install it:

```bash
curl https://sdk.cloud.google.com | bash
```
### 0.1 중요: 쉘 환경 업데이트
설치가 완료된 후, 현재 터미널 창에 변경 사항을 적용하기 위해 아래 명령어를 **반드시** 실행하거나 터미널을 완전히 껐다 켜주세요:

```bash
source ~/.zshrc
```

## 1. Google Cloud Authentication
이제 `gcloud` 명령어가 인식될 것입니다. 아래 명령어로 로그인을 진행해 주세요:
You must authenticate your local environment with Google Cloud. Open your terminal and run:

```bash
gcloud auth login
gcloud auth application-default login
```

## 2. Identify Your Project ID
Find the Google Cloud Project ID where you want to use Stitch. Based on your system, the most relevant one is:
- **`gen-lang-client-0644414376`** (Gemini API)

## 3. Enable Stitch Service
먼저 gcloud 환경에 프로젝트 ID를 고정하고 서비스를 활성화합니다. 아래 두 명령어를 차례대로 터미널에 입력해 주세요:

```bash
gcloud config set project gen-lang-client-0644414376
gcloud beta services mcp enable stitch.googleapis.com
```

## 4. Update the Config File (자동 완료됨)
이미 `~/.gemini/antigravity/mcp_config.json` 파일을 실제 프로젝트 ID(`gen-lang-client-0644414376`)로 제가 직접 수정해 두었습니다. 따로 수정하실 필요 없습니다!

## 5. Verify in Antigravity
Restart Antigravity. Check the MCP status in the bottom right or settings. Once it shows a green light for `stitch`, you can use it!

**Example Usage:**
> @stitch 내 최근 프로젝트의 '로그인 화면' 디자인 정보를 읽어와서 React와 Tailwind CSS 코드로 구현해줘.
