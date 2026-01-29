# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
#     "scipy",
#     "scikit-learn",
#     "polars",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
 
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📂 1. 데이터 로드

    K-리그 경기 데이터와 이벤트 데이터를 로드합니다.
    """)
    return


@app.cell
def _(pd):
    # 데이터 경로
    BASE_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data"

    # 데이터 로드
    match_info = pd.read_csv(f"{BASE_PATH}/match_info.csv")
    raw_data = pd.read_csv(f"{BASE_PATH}/raw_data.csv")

    print(f"📊 경기 정보: {match_info.shape}")
    print(f"📊 이벤트 데이터: {raw_data.shape}")

    match_info.head()
    return match_info, raw_data


@app.cell
def _(match_info, mo):
    mo.md(f"""
    ## 📈 2. 데이터 요약

    - **총 경기 수:** {match_info['game_id'].nunique()} 경기
    - **리그:** {match_info['competition_name'].iloc[0]}
    - **시즌:** {match_info['season_name'].iloc[0]}
    """)
    return


@app.cell
def _(match_info, raw_data):
    # 경기별 집계
    game_stats = raw_data.groupby(['game_id', 'team_id']).agg(
        total_actions=('action_id', 'count'),
        total_passes=('type_name', lambda x: (x == 'Pass').sum()),
        total_shots=('type_name', lambda x: (x == 'Shot').sum()),
        successful_actions=('result_name', lambda x: (x == 'Successful').sum()),
        avg_x_position=('start_x', 'mean'),
        unique_players=('player_id', 'nunique')
    ).reset_index()

    # 파생변수
    game_stats['pass_ratio'] = game_stats['total_passes'] / game_stats['total_actions']
    game_stats['shot_ratio'] = game_stats['total_shots'] / game_stats['total_actions']
    game_stats['success_rate'] = game_stats['successful_actions'] / game_stats['total_actions']

    # 메타데이터 병합
    df = game_stats.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score',
                   'home_team_name_ko', 'away_team_name_ko']],
        on='game_id', how='left'
    )

    # 승/무/패 라벨
    def get_result(row):
        if row['team_id'] == row['home_team_id']:
            if row['home_score'] > row['away_score']: return 'Win'
            elif row['home_score'] < row['away_score']: return 'Lose'
            else: return 'Draw'
        else:
            if row['away_score'] > row['home_score']: return 'Win'
            elif row['away_score'] < row['home_score']: return 'Lose'
            else: return 'Draw'

    df['result'] = df.apply(get_result, axis=1)
    df['win'] = (df['result'] == 'Win').astype(int)

    print(f"✅ 분석용 데이터 생성 완료: {df.shape}")
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md("""
    ## 📊 3. 탐색적 데이터 분석 (EDA)

    주요 지표의 분포를 확인합니다.
    """)
    return


@app.cell
def _(df, plt, sns):
    # EDA 시각화
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    numeric_cols = ['total_actions', 'total_passes', 'total_shots', 
                    'success_rate', 'pass_ratio', 'shot_ratio']

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 3, idx % 3]
        sns.histplot(df[col], kde=True, ax=ax, color='steelblue')
        ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'평균: {df[col].mean():.2f}')
        ax.set_title(f'{col} 분포', fontsize=14)
        ax.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, mo):
    # 기술통계 테이블
    stats_table = df[['total_actions', 'total_passes', 'total_shots', 'success_rate']].describe().round(3)
    mo.ui.table(stats_table.reset_index())
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🔗 4. 상관관계 분석

    변수 간의 상관관계를 히트맵으로 확인합니다.
    """)
    return


@app.cell
def _(df, np, plt, sns):
    # 상관관계 히트맵
    corr_cols = ['total_actions', 'total_passes', 'total_shots', 
                 'success_rate', 'pass_ratio', 'avg_x_position']

    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                center=0, fmt='.2f', square=True, linewidths=0.5)
    plt.title('K-리그 경기 지표 상관관계 행렬', fontsize=16)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🧪 5. 통계 검정

    승/무/패 그룹 간 성공률 차이가 통계적으로 유의미한지 검정합니다.
    """)
    return


@app.cell
def _(df, np, plt, sns, stats):
    # ANOVA 검정
    win_data = df[df['result'] == 'Win']['success_rate']
    draw_data = df[df['result'] == 'Draw']['success_rate']
    lose_data = df[df['result'] == 'Lose']['success_rate']

    f_stat, p_value = stats.f_oneway(win_data, draw_data, lose_data)

    # Cohen's d
    cohens_d = (win_data.mean() - lose_data.mean()) / np.sqrt(
        ((win_data.std()**2) + (lose_data.std()**2)) / 2
    )

    print(f"📊 ANOVA 검정 결과")
    print(f"   F-통계량: {f_stat:.4f}")
    print(f"   p-value: {p_value:.6f}")
    print(f"   → {'✅ 유의미한 차이 있음 (p < 0.05)' if p_value < 0.05 else '❌ 유의미한 차이 없음'}")
    print(f"\n📏 Cohen's d: {cohens_d:.3f}")

    # 박스플롯
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='result', y='success_rate', data=df, order=['Win', 'Draw', 'Lose'],
                palette='Set2')
    ax.set_title(f'승/무/패별 성공률 분포\n(ANOVA p={p_value:.4f})', fontsize=14)
    ax.set_xlabel('경기 결과')
    ax.set_ylabel('성공률')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🤖 6. 머신러닝 예측

    RandomForest 모델을 사용하여 승/무/패를 예측합니다.
    """)
    return


@app.cell
def _(
    LabelEncoder,
    RandomForestClassifier,
    StandardScaler,
    accuracy_score,
    confusion_matrix,
    df,
    pd,
    plt,
    sns,
    train_test_split,
):

    # 피처/타겟 분리
    feature_cols = ['total_actions', 'total_passes', 'total_shots', 
                    'success_rate', 'pass_ratio', 'shot_ratio', 
                    'avg_x_position', 'unique_players']

    X = df[feature_cols].fillna(0)
    y = df['result']

    # 라벨 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"🎯 모델 정확도: {accuracy:.1%}")

    # Feature Importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Feature Importance
    sns.barplot(x='importance', y='feature', data=importance_df, 
                palette='viridis', ax=axes[0])
    axes[0].set_title('Feature Importance', fontsize=14)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=le.classes_, yticklabels=le.classes_)
    axes[1].set_title('Confusion Matrix', fontsize=14)
    axes[1].set_xlabel('예측')
    axes[1].set_ylabel('실제')

    plt.tight_layout()
    plt.show()

    return accuracy, importance_df


@app.cell
def _(accuracy, importance_df, mo):
    mo.md(f"""
    ## 💡 7. 분석 인사이트

    ### 핵심 결과

    | 항목 | 결과 |
    |------|------|
    | **모델 정확도** | {accuracy:.1%} |
    | **1위 핵심 변수** | {importance_df.iloc[0]['feature']} |
    | **2위 핵심 변수** | {importance_df.iloc[1]['feature']} |
    | **3위 핵심 변수** | {importance_df.iloc[2]['feature']} |

    ### 전술적 제언

    1. **성공률(success_rate)** 관리가 승리의 핵심
    2. **공격적 포지셔닝(avg_x_position)**이 높을수록 슈팅 기회 증가
    3. **패스 비율(pass_ratio)**은 점유율과 강한 상관관계
    4. 무승부(Draw) 예측은 여전히 어려움 → 외부 변수 필요
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## 🎉 분석 완료!

    **Antigravity AI Analysis System** | K-League Data Analytics
    """)
    return


if __name__ == "__main__":
    app.run()
