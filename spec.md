# K-League Win Rate Analysis Dashboard Spec

## 1. 개요 (Overview)
*   **프로젝트명**: K-League Team Win Rate Dashboard
*   **목표**: K리그 구단별 승률을 분석하고 대시보드로 시각화하여 팀 간 성과를 비교한다.
*   **타겟 독자**: 구단 관계자, 축구 팬, 데이터 분석가
*   **도구**: Python, Streamlit, Pandas, Plotly, DuckDB (전처리)

## 2. 데이터 요구사항 (Data Requirements)
*   **필수 데이터**: `match_info.csv` (경기 결과 데이터)
    *   `home_team_name`, `away_team_name` (팀명)
    *   `home_score`, `away_score` (점수 -> 승패 판별용)
    *   `season_id` (시즌별 트렌드 분석용)
*   **전처리 로직**:
    *   경기별 승/무/패 파생변수 생성 (`Win`, `Draw`, `Loss`)
    *   홈/원정 경기 통합하여 팀 기준 데이터셋 재구성

## 3. 기능 명세 (Feature Specifications)

### 3.1. 대시보드 구조 (Streamlit Layout)
*   **제목**: `🏆 K리그 구단별 승률 분석 대시보드`
*   **사이드바**:
    *   `시즌 선택` (Multi-select: 전체 보기 or 특정 시즌)
    *   `팀 선택` (Multi-select: 특정 팀 간 비교)
*   **메인 패널**:
    1.  **KPI 카드**: 전체 평균 승률, 최고 승률 팀, 최저 승률 팀
    2.  **승률 랭킹 차트 (Bar/Column Chart)**:
        *   X축: 팀명 (승률 순 정렬)
        *   Y축: 승률 (%)
        *   툴팁: 승/무/패 경기 수 상세 표시
    3.  **승/무/패 비중 차트 (Stacked Bar Chart)**:
        *   팀별 전적(승/무/패) 비율 시각화
    4.  **Raw Data 테이블**: 필터링된 데이터 원본 보기 (검색 기능 포함)

### 3.2. 분석 로직 (Analysis Logic)
*   **승률 계산 공식**:
    $$ \text{Win Rate} = \frac{\text{Wins}}{\text{Total Games}} \times 100 $$
*   (옵션) 승점 기반 순위 계산 로직 ($$ \text{Points} = \text{Win} \times 3 + \text{Draw} \times 1 $$)

## 4. 실행 계획 (Execution Plan) - Chunking
이 프로젝트는 다음 청크 단위로 나누어 구현한다.

*   **[Chunk 1] 데이터 로드 및 전처리**:
    *   `match_info.csv` 로드
    *   팀 관점으로 데이터 재구조화 (Wide to Long)
    *   승/무/패 컬럼 생성
*   **[Chunk 2] 통계 집계 로직 구현**:
    *   팀별 총 경기 수, 승, 무, 패, 승률 계산 함수 작성
*   **[Chunk 3] Streamlit 대시보드 구현**:
    *   레이아웃 잡기 (사이드바, 메인)
    *   Plotly 차트 연동
*   **[Chunk 4] 테스트 및 배포**:
    *   데이터 정합성 검증
    *   `🚀_분석도구_실행하기.command`에 연결

## 5. 규칙 및 컨벤션 (GEMINI.md 준수)
*   모든 코드는 `GEMINI.md`의 스타일 가이드를 따른다.
*   시각화는 색맹 친화적인 팔레트 사용 (Plotly 기본 또는 커스텀)
*   한글 폰트 깨짐 방지 처리 필수.
