# 📱 EPL-X Manager 모바일 앱 스토어 배포 전략 가이드

작성자: Senior Analyst Antigravity (2026.01.15)
목표: Python Streamlit 웹앱을 Apple App Store 및 Google Play Store에 정식 배포

---

## 📅 Roadmap 요약
1. **GitHub 배포**: 현재 코드를 GitHub에 업로드 (Code Base)
2. **Cloud 호스팅**: Streamlit Community Cloud (무료)에 연결하여 `https://epl-x-manager.streamlit.app` 형태의 URL 획득
3. **App Wrapping**: 해당 URL을 열어주는 "껍데기 앱(Native Wrapper)" 제작
4. **Store 심사**: 개발자 계정 등록 및 심사 제출

---

## 🛠 Phase 1: 클라우드 호스팅 (필수 선행)
앱 스토어에 올리려면 앱이 24시간 돌아가는 "서버 링크"가 필요합니다.

1. **GitHub 업로드**: 현재 로컬 폴더를 GitHub 리포지토리에 푸시합니다.
2. **Streamlit Cloud 연결**:
   - [share.streamlit.io](https://share.streamlit.io) 접속 및 가입
   - 'New App' -> GitHub 리포지토리 선택 -> `epl_project/app.py` 지정
   - **Deploy!** (URL이 생성됩니다. 예: `https://my-epl-app.streamlit.app`)

   > **💡 팁**: 이 URL이 생겨야 다음 단계인 앱 변환이 가능합니다.

---

## 📲 Phase 2: 모바일 앱 패키징 (Webview Strategy)
코딩 없이 URL만으로 앱 설치 파일(.apk, .ipa)을 만드는 방법입니다.

### 방법 A: 전문 변환 서비스 이용 (추천: 가장 빠름)
비용을 조금 지불하더라도 확실하게 스토어용 파일을 받는 방법입니다.
*   **Median.co (구 GoNative)**: URL만 넣으면 안드로이드/iOS 앱 소스를 줍니다.
*   **WebViewGold**: $50 정도에 템플릿을 구매하여 URL만 바꿔치기 하는 방식입니다.

### 방법 B: 직접 패키징 (무료, 개발 지식 필요)
*   **Flutter / React Native**: "Webview 라이브러리" 하나만 써서 앱을 빌드합니다.
*   **장점**: 완전 무료, 네이티브 기능(푸시 알림 등) 추가 가능.
*   **단점**: 맥(Mac)에 Xcode 설치 필수, 빌드 환경 세팅이 복잡함.

---

## 👮‍♂️ Phase 3: 스토어 심사 통과 전략 (중요!)
애플은 단순 웹사이트를 앱으로 올리는 것을 싫어합니다 (Guideline 4.2). 이를 돌파하기 위한 전략입니다.

### 1. Apple App Store (까다로움)
*   **4.2.2 Minimum Functionality 방어 논리**:
    *   "이 앱은 단순 정보 제공 웹사이트가 아니다."
    *   "AI 예측 시뮬레이션 기능, 실시간 알림, 방대한 데이터베이스와의 상호작용은 네이티브 앱 환경에서 최적화된다." 라고 소명해야 합니다.
    *   **Native Navigation**: 하단 탭바(Tab Bar)를 네이티브 코드(Swift/Wrapper 설정)에서 구현해주면 승인 확률이 급격히 올라갑니다.

### 2. Google Play Store (수월함)
*   구글은 웹뷰 앱에 관대합니다. 기본적인 보안 정책만 지키면 통과됩니다.

---

## 💰 준비물 및 비용
| 항목 | 비용 | 비고 |
| :--- | :--- | :--- |
| **GitHub 계정** | 무료 | 소스코드 저장소 |
| **Streamlit Cloud** | 무료 | 서버 호스팅 |
| **Apple 개발자 계정** | $99/년 | 앱스토어 등록 필수 |
| **Google 개발자 계정** | $25 (1회) | 플레이스토어 등록 필수 |
| **Median.co (선택)** | 무료~유료 | 앱 변환 도구 |

---

## 🚀 다음 실행 단계 (Action Item)
1. 지금 즉시 이 폴더를 **GitHub**에 올리세요.
2. **Streamlit Cloud**에 배포하여 URL을 확보하세요.
3. URL이 나오면 저에게 알려주세요. 그 다음 **"앱 아이콘 디자인"**과 **"스크린샷 생성"**을 도와드리겠습니다.
