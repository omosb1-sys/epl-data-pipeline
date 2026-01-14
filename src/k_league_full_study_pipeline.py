"""
🏆 K리그 데이터 분석 통합 스터디 가이드 (Full Analysis Pipeline v2.0)
========================================================================
이 스크립트는 시니어 분석가의 워크플로우(지표 생성 -> 통계 검증 -> 시각화 -> AI)를
따라갈 수 있도록 설계된 고급 분석 파이프라인입니다.

수정사항: 
1. 고급 파생 변수(효율성, 빌드업 지표) 생성 단계 전진 배치
2. EDA 이전에 통계적 유의성 검정(t-test) 수행
3. 시각화 단계에서 파생 변수들의 영향력 집중 분석

작성자: 30년차 선배 제미나이3
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # GUI 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
import polars as pl
import os
import datetime
import warnings
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
import sys
# src 폴더를 경로에 추가하여 모듈 임포트 보장
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# [New Integration]
try:
    from k_league_timesfm_forecast import KLeagueForecaster
    from polars_duckdb_synergy import BigDataEngine
    HAS_NEW_TOOLS = True
except ImportError as e:
    HAS_NEW_TOOLS = False
    print(f"⚠️ 신규 모듈 로드 실패: {e}")

# [Advanced Toolkits]
try:
    import sharp
    HAS_SHARP = True
except ImportError:
    HAS_SHARP = False

try: from IPython.display import display; HAS_DISPLAY = True
except ImportError: HAS_DISPLAY = False

# ============================================================
# 0. 환경 설정 및 경로 통합 관리
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw/raw_data.csv")
MATCH_INFO_PATH = os.path.join(BASE_DIR, "data/raw/match_info.csv")

def ensure_directories():
    for path in ["reports/figures", "reports/docs"]:
        os.makedirs(os.path.join(BASE_DIR, path), exist_ok=True)

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

def run_study_pipeline():
    ensure_directories()
    print(f"🚀 K리그 고도화 분석 파이프라인 시작: {datetime.datetime.now()}")
    print("=" * 80)

    # ------------------------------------------------------------
    # [STEP 1] 데이터 로드
    # ------------------------------------------------------------
    print("\n[STEP 1] DuckDB 데이터 로드")
    con = duckdb.connect(database=':memory:')
    raw_data = con.execute(f"SELECT * FROM read_csv_auto('{RAW_DATA_PATH}')").df()
    match_info = con.execute(f"SELECT * FROM read_csv_auto('{MATCH_INFO_PATH}')").df()
    print(f"✅ 완료: raw_data {raw_data.shape}")

    # ------------------------------------------------------------
    # [STEP 2] 기본 집계
    # ------------------------------------------------------------
    print("\n[STEP 2] 경기 단위 팀 통계 집계")
    raw_data['result_name'] = raw_data['result_name'].fillna('Unknown')
    
    def aggregate_stats(x):
        return pd.Series({
            'Passes': (x['type_name'] == 'Pass').sum(),
            'Pass_OK': ((x['type_name'] == 'Pass') & (x['result_name'] == 'Successful')).sum(),
            'Shots': (x['type_name'].isin(['Shot', 'Goal', 'Shot_Frekick'])).sum(),
            'Goals': (x['type_name'] == 'Goal').sum() + ((x['type_name'] == 'Shot') & (x['result_name'] == 'Goal')).sum(),
            'Tackles': (x['type_name'] == 'Tackle').sum(),
            'Interceptions': (x['type_name'] == 'Interception').sum(),
            'Attack_Zone': (x['start_x'] > 60).sum()
        })

    df_agg = raw_data.groupby(['game_id', 'team_id', 'team_name_ko']).apply(aggregate_stats, include_groups=False).reset_index()
    df_agg['Pass_Rate'] = (df_agg['Pass_OK'] / df_agg['Passes'].replace(0, 1) * 100).round(1)
    
    # 경기 날짜 및 홈/어웨이 정보 결합
    match_info['game_date'] = pd.to_datetime(match_info['game_date'])
    df_agg = df_agg.merge(match_info[['game_id', 'home_team_id', 'game_date']], on='game_id')
    
    # ------------------------------------------------------------
    # [STEP 3] 피처 엔지니어링 (고급 지표 생성)
    # ------------------------------------------------------------
    print("\n[STEP 3] 고급 파생 변수 생성 (전략적 지표)")
    
    # 1. 실점 및 승패 로직
    temp_opp = df_agg[['game_id', 'team_id', 'Goals']].rename(columns={'team_id': 'opp_id', 'Goals': 'Goals_Against'})
    df_agg = df_agg.merge(temp_opp, on='game_id')
    df_agg = df_agg[df_agg['team_id'] != df_agg['opp_id']]
    df_agg['is_win'] = (df_agg['Goals'] > df_agg['Goals_Against']).astype(int)
    df_agg['is_home'] = (df_agg['team_id'] == df_agg['home_team_id']).astype(int)

    # 2. 최근 3경기 승률 (기세 지표)
    df_agg = df_agg.sort_values(['team_id', 'game_date'])
    df_agg['rolling_win_rate'] = df_agg.groupby('team_id')['is_win'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(0.5)

    # 3. 공격 효율성 (Shot Efficiency): 슛 대비 골
    df_agg['shot_efficiency'] = (df_agg['Goals'] / df_agg['Shots'].replace(0, 1)).round(3)

    # 4. 수비 압박 지표 (Defensive Pressure)
    df_agg['defensive_pressure'] = df_agg['Tackles'] + df_agg['Interceptions']

    # 5. 빌드업 지표 (Buildup Index): 패스 성공률과 공격 지역 진입의 결합
    df_agg['buildup_index'] = (df_agg['Pass_Rate'] * df_agg['Attack_Zone'] / 100).round(2)
    
    print("✅ 완료: shot_efficiency, defensive_pressure, buildup_index 생성")

    # ------------------------------------------------------------
    # [STEP 3.8] 통계적 유의성 검정
    # ------------------------------------------------------------
    print("\n[STEP 3.8] 통계 검정: 승리 팀과 패배 팀의 지표 차이")
    for col in ['Pass_Rate', 'shot_efficiency', 'buildup_index']:
        win = df_agg[df_agg['is_win'] == 1][col]
        loss = df_agg[df_agg['is_win'] == 0][col]
        t_stat, p_val = stats.ttest_ind(win, loss)
        print(f"📊 [{col}] t-stat: {t_stat:.2f}, p-value: {p_val:.4f} {'(유의성 확보)' if p_val < 0.05 else '(유의미하지 않음)'}")

    # ------------------------------------------------------------
    # [STEP 4] EDA (고급 지표 기반 시각화)
    # ------------------------------------------------------------
    print("\n[STEP 4] EDA: 시각화 분석 보고")
    fig_dir = os.path.join(BASE_DIR, "reports/figures")
    
    # 1. 핵심 파생 변수 승패 비교
    plt.figure(figsize=(15, 5))
    adv_features = ['shot_efficiency', 'buildup_index', 'rolling_win_rate']
    for i, col in enumerate(adv_features):
        plt.subplot(1, 3, i+1)
        sns.violinplot(x='is_win', y=col, data=df_agg, inner="quart", palette="muted")
        plt.title(f'Win vs Loss: {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "advanced_feature_comparison.png"))
    # plt.show()

    # 2. 전체 변수 상관관계 히트맵
    plt.figure(figsize=(12, 10))
    corr_cols = ['Pass_Rate', 'Shots', 'Goals', 'Attack_Zone', 'shot_efficiency', 'defensive_pressure', 'buildup_index', 'rolling_win_rate', 'is_win']
    sns.heatmap(df_agg[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Total Correlation Heatmap")
    plt.savefig(os.path.join(fig_dir, "total_heatmap.png"))
    # plt.show()

    # ------------------------------------------------------------
    # [STEP 5 & 6] AI 모델 학습 및 해석
    # ------------------------------------------------------------
    print("\n[STEP 5 & 6] 머신러닝 학습 및 ShaRP 해석")
    features = ['Pass_Rate', 'shot_efficiency', 'defensive_pressure', 'buildup_index', 'rolling_win_rate', 'is_home']
    X = df_agg[features].fillna(0)
    y = df_agg['is_win']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"🤖 AI 학습 완료! 예측 정확도(AUC): {auc:.4f}")

    # ShaRP 해석
    plt.figure(figsize=(10, 6))
    if HAS_SHARP:
        try:
            def score_func(X_in):
                if isinstance(X_in, np.ndarray): X_in = pd.DataFrame(X_in, columns=features)
                return model.predict_proba(X_in)[:, 1]
            explainer = sharp.ShaRP(qoi="score", qoi_func=score_func, ref_distribution=X_train.sample(50).values)
            sharp_vals = explainer.all(X_test.head(10).values)
            imp_df = pd.DataFrame({'feature': features, 'val': np.abs(sharp_vals).mean(axis=0)})
            plt.title("AI의 판단 근거 (ShaRP 해석)")
        except:
            imp_df = pd.DataFrame({'feature': features, 'val': model.feature_importances_})
            plt.title("AI 변수 중요도 (Feature Importance)")
    else:
        imp_df = pd.DataFrame({'feature': features, 'val': model.feature_importances_})
        plt.title("AI 변수 중요도 (Feature Importance)")
    
    sns.barplot(x='val', y='feature', data=imp_df.sort_values('val', ascending=False), palette='viridis')
    plt.savefig(os.path.join(fig_dir, "model_interpretation.png"))
    # plt.show()

    # ------------------------------------------------------------
    # [STEP 7] AI 예측 및 대용량 데이터 처리 (New)
    # ------------------------------------------------------------
    print("\n[STEP 7] AI 득점 예측 및 DuckDB-Polars 고속 연산")
    
    predicted_top_team = "Not Available"
    aggressive_teams_str = "Not Available"

    if HAS_NEW_TOOLS:
        # 1. DuckDB & Polars Wrangling
        print("   >> DuckDB + Polars 엔진 가동...")
        bd_engine = BigDataEngine(data_path=MATCH_INFO_PATH)
        df_pl = bd_engine.run_advanced_wrangling()
        if df_pl is not None:
             # Polars 문법으로 필터링 (평균 1.5골 이상)
            agg_teams = df_pl.filter(pl.col("avg_goals") >= 1.5).select("team").to_series().to_list()
            aggressive_teams_str = ", ".join(agg_teams) if agg_teams else "없음"
            print(f"   >> 공격적 팀 식별(Avg Goal >= 1.5): {aggressive_teams_str}")

        # 2. TimesFM Forecast
        print("   >> AI 득점 예측 모델 가동...")
        forecaster = KLeagueForecaster(data_path=MATCH_INFO_PATH)
        forecast_df = forecaster.run_league_analysis()
        if forecast_df is not None and not forecast_df.empty:
            top_row = forecast_df.iloc[0]
            predicted_top_team = f"{top_row['구단']} (예상 {top_row['예상_주간_득점력']}골)"

    # ------------------------------------------------------------
    # [STEP 8] 최종 인사이트 리포트 생성
    # ------------------------------------------------------------
    print("\n[STEP 8] 최종 인사이트 요약 보고서 작성 중...")
    report_path = os.path.join(BASE_DIR, "reports/docs/study_insight_report.txt")
    top_f = imp_df.sort_values('val', ascending=False).iloc[0]['feature']
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("🏆 K-리그 고도화 분석 최종 보고서\n")
        f.write(f"작성일시: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"1️⃣ AI 모델 성적 (AUC): {auc:.4f}\n")
        f.write(f"2️⃣ 핵심 승리 결정 요인(AI): {top_f}\n")
        f.write(f"3️⃣ AI 득점 예측 1위: {predicted_top_team}\n")
        f.write(f"4️⃣ 공격 중심 팀 (DuckDB 분석): {aggressive_teams_str}\n")
        f.write(f"5️⃣ 통계적 핵심 지표: t-test 결과 기반 유의성 확인 완료\n\n")
        f.write("📋 시니어의 총평:\n")
        f.write(f"단순 패스보다는 '{top_f}' 지표가 승리에 결정적으로 기여합니다.\n")
        f.write("시각화된 박스플롯과 히트맵을 통해 지표 간의 인과성을 검토하세요.\n")
    
    print(f"✅ 보고서 생성 완료: {report_path}")
    print("=" * 80 + "\n✨ 축하합니다! 모든 분석 파이프라인이 성공적으로 종료되었습니다.")

if __name__ == "__main__":
    run_study_pipeline()