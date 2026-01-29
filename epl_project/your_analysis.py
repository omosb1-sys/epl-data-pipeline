import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    # 🎨 [시니어 분석가 특제] 초대형 UI 세팅
    mo.md("""
    <style>
        :root {
            font-size: 130% !important;
        }
        .marimo-cell {
            margin-bottom: 3rem !important;
        }
        h1 { font-size: 3.2rem !important; color: #1f77b4; margin-bottom: 1.5rem; }
        h3 { font-size: 2.2rem !important; margin-top: 1rem; }

        /* 차트 및 테이블 텍스트 크기 강제 확대 */
        canvas {
            zoom: 1.3;
        }
        .marimo-table {
            font-size: 1.2rem !important;
        }
    </style>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # ⚽ EPL 팀별 득점 정밀 분석 대시보드
    사용자님이 질문하신 **f-string** 문법을 활용한 실시간 리포트입니다.
    """)
    return


@app.cell
def _(duckdb):
    # 💾 데이터 로드
    db_path = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data/epl_analytics.duckdb"
    conn = duckdb.connect(db_path)

    _query = """
    WITH team_goals AS (
        SELECT home_team AS team, home_score AS goals FROM match_logs
        UNION ALL
        SELECT away_team AS team, away_score AS goals FROM match_logs
    )
    SELECT team, CAST(SUM(goals) AS INTEGER) as total_goals, COUNT(*) as games_played
    FROM team_goals
    GROUP BY team
    ORDER BY total_goals DESC
    """
    df_goals = conn.execute(_query).df()
    conn.close()
    return (df_goals,)


@app.cell
def _(df_goals, mo):
    # 🎛️ 슬라이더 설정
    max_val = int(df_goals["total_goals"].max()) if not df_goals.empty else 10

    score_slider = mo.ui.slider(
        start=0, 
        stop=max_val, 
        step=1, 
        value=0, 
        label="🏆 최소 득점 필터"
    )

    # 💡 f-string과 mo.md를 결합한 예시입니다!
    # 주피터의 print() 대신 이렇게 쓰시면 화면에 실시간으로 반영됩니다.
    mo.md(f"### 🔍 현재 필터 기준: **{score_slider.value}점** 이상\n{score_slider}")
    return (score_slider,)


@app.cell
def _(alt, df_goals, score_slider):
    # 📊 데이터 필터링 
    _limit = score_slider.value
    filtered_df = df_goals[df_goals["total_goals"] >= _limit].copy()

    # 차트 객체 생성
    chart = alt.Chart(filtered_df).mark_bar(color='#2ca02c').encode(
        x=alt.X('team:N', sort='-y', title='팀명', axis=alt.Axis(labelFontSize=14)),
        y=alt.Y('total_goals:Q', title='총 득점', axis=alt.Axis(labelFontSize=14)),
        tooltip=['team', 'total_goals']
    ).properties(
        width='container',
        height=400
    )
    return chart, filtered_df


@app.cell
def _(chart, mo):
    # 🚀 차트와 텍스트를 함께 출력 (mo.md에서 f-string 활용)
    mo.vstack([
        mo.md("### 📈 팀별 화력 분포 차트"),
        chart
    ])
    return


@app.cell
def _(filtered_df, mo):
    # 📑 테이블과 요약 정보를 함께 출력
    summary_msg = f"현재 **{len(filtered_df)}개**의 팀이 조건을 만족합니다."

    mo.vstack([
        mo.md(f"### 📑 상세 데이터\n{summary_msg}"),
        mo.ui.table(filtered_df)
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
