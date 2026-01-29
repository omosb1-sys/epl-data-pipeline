import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import pandas as pd
    return duckdb, mo, pd


@app.cell
def _(mo):
    # 전반적인 UI 스케일을 키우는 CSS 주입
    mo.md(
        """
        <style>
            :root {
                font-size: 115%; /* 전체적인 텍스트 및 UI 비율 15% 증대 */
            }
            .marimo-cell {
                margin-bottom: 2rem !important; /* 셀 간격 확장 */
            }
            button, input, select {
                padding: 10px 15px !important; /* 조작 버튼 크기 확장 */
            }
            h1 { font-size: 2.5rem !important; }
            h2 { font-size: 2rem !important; }
            h3 { font-size: 1.5rem !important; }
        </style>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # ⚽ Antigravity x marimo: EPL 정밀 분석기
    이 대시보드는 **마리모의 리액티브 엔진**과 **안티그래비티의 분석 로직**이 결합된 결과물입니다.
    """)
    return


@app.cell
def _(duckdb):
    # DuckDB 연결 및 데이터 로드
    # 데이터베이스 파일 존재 여부를 확인하여 에러 방지
    db_path = '/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data/epl_analytics.duckdb'
    conn = duckdb.connect(db_path)
    df = conn.execute("SELECT * FROM match_logs").df()
    return (df,)


@app.cell
def _(mo):
    # 마리모 위젯: 득점 하한선 슬라이더
    score_threshold = mo.ui.slider(start=0, stop=5, step=1, value=2, label="최소 득점 필터")
    mo.md(f"### 📊 분석 조건 설정: {score_threshold}")
    return (score_threshold,)


@app.cell
def _(df, score_threshold):
    # 슬라이더 값에 따라 실시간 필터링 (Reactive!)
    filtered_df = df[df['home_score'] >= score_threshold.value]
    return (filtered_df,)


@app.cell
def _(filtered_df, mo):
    # 결과 출력
    has_data = filtered_df is not None and not filtered_df.empty
    avg_xg = f"{filtered_df['xG_home'].mean():.2f}" if has_data else "0.00"

    mo.hstack([
        mo.stat(label="필터링된 경기 수", value=len(filtered_df) if has_data else 0),
        mo.stat(label="평균 xG(홈)", value=avg_xg)
    ], justify="start")
    return


@app.cell
def _(filtered_df, mo):
    # 인터랙티브 테이블
    mo.ui.table(filtered_df)
    return


if __name__ == "__main__":
    app.run()
