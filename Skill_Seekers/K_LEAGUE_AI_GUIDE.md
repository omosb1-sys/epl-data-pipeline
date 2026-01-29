# 🤖 K-리그 AI 분석 시스템 사용 가이드

**Gemini 2.0 Flash 기반 실시간 전문가급 인사이트 제공**

---

## 🎯 개요

이 시스템은 **Gemini API**를 활용하여 K-리그 데이터를 30년 차 시니어 분석가의 시각으로 해석합니다.

### 주요 기능

1. **팀 성적 AI 분석** - 특정 팀의 심층 분석 및 인사이트
2. **라이벌 매치 예측** - 두 팀 간의 승부 예측 및 전술 분석
3. **리그 전체 트렌드** - 시즌 전체 흐름 및 파워 랭킹
4. **기본 통계 분석** - API 키 없이도 사용 가능한 기초 분석

---

## 🚀 빠른 시작

### 1. Gemini API 키 발급

1. https://makersuite.google.com/app/apikey 접속
2. "Create API Key" 클릭
3. 생성된 API 키 복사

### 2. 환경변수 설정

```bash
# 터미널에서 실행 (임시)
export GEMINI_API_KEY="your-api-key-here"

# 영구 설정 (권장)
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. 대시보드 실행

**방법 1: 실행 스크립트 (권장)**
```bash
# Finder에서 더블클릭
🚀_AI_분석_대시보드_실행하기.command
```

**방법 2: 터미널 명령어**
```bash
cd /Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터
streamlit run src/app_kleague_ai.py
```

### 4. 브라우저 접속

자동으로 브라우저가 열리며, 수동 접속 시:
```
http://localhost:8501
```

---

## 📊 사용 방법

### 기본 통계 분석 (API 키 불필요)

1. 사이드바에서 **"📈 기본 통계"** 선택
2. 분석 주제 선택:
   - 팀별 실점 분석
   - 홈/어웨이 득점 비교
   - 시간대별 골 분포

### AI 심층 분석 (API 키 필요)

1. 사이드바에서 **"🤖 AI 심층 분석"** 선택
2. 분석 주제 선택:

#### 🆕 팀 성적 AI 분석
- 분석할 팀 선택
- "🚀 AI 분석 시작" 버튼 클릭
- 약 3~5초 후 전문가급 리포트 생성
- "💾 리포트 저장" 버튼으로 마크다운 파일 저장

**분석 구조:**
```
🎯 결론 (Conclusion)
- 한 문장 핵심 진단

📊 근거 (Evidence)
- 통계적 근거 3가지
- 최근 흐름 분석

💡 제언 (Recommendation)
- 전술적 개선 방향
- Shadow KPI 제안

Why Now?
- 현재 시점의 맥락 설명
```

#### 🆕 라이벌 매치 예측
- 대결할 두 팀 선택
- "🔮 승부 예측" 버튼 클릭
- 승부 예측 및 전술 분석 제공

**분석 구조:**
```
⚔️ 승부 예측
- 예상 승자 및 확률

🎯 전술적 우위
- 각 팀의 강점 분석

🔍 승부처 (Key Battle)
- 경기를 결정할 핵심 요소
```

#### 🆕 리그 전체 트렌드
- "🚀 리그 분석 시작" 버튼 클릭
- 전체 리그 파워 랭킹 및 트렌드 분석

**분석 구조:**
```
🏆 파워 랭킹 Top 3
⚠️ 위기의 팀 Bottom 3
📈 리그 트렌드
🔮 시즌 전망
```

---

## 🛠️ 고급 사용법

### Python 스크립트에서 직접 사용

```python
from src.gemini_k_league_analyst import GeminiKLeagueAnalyst
import pandas as pd

# 분석가 초기화
analyst = GeminiKLeagueAnalyst(api_key="your-api-key")

# 데이터 로드
df = pd.read_csv("data/k_league_2024.csv")

# 팀 분석
result = analyst.analyze_team_performance(df, "전북 현대")
print(result['analysis'])

# 리포트 저장
filepath = analyst.save_report(result)
print(f"리포트 저장: {filepath}")
```

### 배치 분석 (모든 팀 자동 분석)

```python
import pandas as pd
from src.gemini_k_league_analyst import GeminiKLeagueAnalyst

analyst = GeminiKLeagueAnalyst()
df = pd.read_csv("data/k_league_2024.csv")

teams = df['팀명'].unique()

for team in teams:
    print(f"분석 중: {team}")
    result = analyst.analyze_team_performance(df, team)
    filepath = analyst.save_report(result, output_dir="reports/batch")
    print(f"✅ {team} 리포트 저장: {filepath}")
```

---

## 🎨 커스터마이징

### 분석 페르소나 변경

`src/gemini_k_league_analyst.py` 파일의 `system_prompt` 수정:

```python
self.system_prompt = """
당신은 [원하는 페르소나]입니다.

**핵심 원칙:**
1. [원칙 1]
2. [원칙 2]
...
"""
```

### 추가 분석 기능 구현

```python
def custom_analysis(self, df: pd.DataFrame, param: str) -> Dict[str, str]:
    """커스텀 분석 함수"""
    
    # 데이터 전처리
    processed_data = self._preprocess(df, param)
    
    # 프롬프트 생성
    prompt = f"""
    {self.system_prompt}
    
    [커스텀 분석 요청]
    {processed_data}
    """
    
    # Gemini 호출
    response = self.model.generate_content(prompt)
    
    return {
        "analysis": response.text,
        "timestamp": datetime.now().isoformat()
    }
```

---

## 🔧 문제 해결

### API 키 오류
```
❌ GEMINI_API_KEY가 필요합니다.
```

**해결 방법:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 데이터베이스 없음
```
❌ 데이터베이스를 찾을 수 없습니다
```

**해결 방법:**
```bash
# 데이터베이스 경로 확인
ls -la data/processed/kleague.db

# 없다면 데이터 초기화 스크립트 실행
python src/init_db.py
```

### Streamlit 포트 충돌
```
OSError: [Errno 48] Address already in use
```

**해결 방법:**
```bash
# 다른 포트로 실행
streamlit run src/app_kleague_ai.py --server.port 8502
```

### Gemini API 할당량 초과
```
Error: Quota exceeded
```

**해결 방법:**
- 무료 할당량: 분당 60회 요청
- 대기 후 재시도 또는 유료 플랜 업그레이드

---

## 📈 성능 최적화

### 응답 속도 개선

1. **캐싱 활용**
```python
@st.cache_data(ttl=3600)  # 1시간 캐시
def cached_analysis(team_name: str):
    return analyst.analyze_team_performance(df, team_name)
```

2. **배치 처리**
```python
# 여러 팀을 한 번에 분석
teams = ["전북 현대", "울산 현대", "포항 스틸러스"]
results = [analyst.analyze_team_performance(df, t) for t in teams]
```

### 비용 절감

1. **로컬 캐시 활용** - 동일 분석 재요청 방지
2. **요약 프롬프트** - 불필요한 장문 응답 제한
3. **배치 분석** - 한 번에 여러 팀 분석

---

## 🤝 기여 및 피드백

### 버그 리포트
- GitHub Issues 또는 이메일로 제보

### 기능 제안
- 새로운 분석 기능 아이디어 환영

### 코드 기여
- Pull Request 환영 (GEMINI.md Protocol 준수)

---

## 📚 참고 자료

- [Gemini API 문서](https://ai.google.dev/docs)
- [Streamlit 문서](https://docs.streamlit.io)
- [GEMINI.md Protocol](../GEMINI.md)

---

## 📄 라이선스

MIT License

---

**🤖 Powered by Gemini 2.0 Flash (Experimental)**  
**📖 GEMINI.md Protocol v1.9**  
*Developed by Antigravity AI*

---

## 🎉 다음 단계

1. ✅ 기본 사용법 숙지
2. ✅ 첫 번째 팀 분석 실행
3. ✅ 리포트 저장 및 공유
4. 🚀 커스텀 분석 기능 추가
5. 🚀 다른 리그 데이터 통합

**Happy Analyzing! ⚽🤖**
