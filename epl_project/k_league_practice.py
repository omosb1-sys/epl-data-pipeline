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
    # ⚽ K-리그 데이터 분석 연습

    자유롭게 코드를 작성하세요!
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_squared_error
    from scipy.signal import savgol_filter
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.proportion import proportions_ztest
    from datetime import datetime
    import warnings
    import os
    import json

    warnings.filterwarnings('ignore')

    # ============================================
    # 0. 환경 설정
    # ============================================
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-whitegrid')

    BASE_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data"
    OUTPUT_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/output"
    REPORT_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/reports"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(REPORT_PATH, exist_ok=True)

    # 분석 결과 저장용
    ANALYSIS_RESULTS = {
        'meta': {},
        'eda': {},
        'general_stats': {},  # 일반 통계분석 결과
        'statistics': {},
        'ml': {},
        'causal': {},
        'timeseries': {},
        'insights': []
    }


    def print_header(title: str, emoji: str = "📊"):
        """섹션 헤더 출력"""
        print("\n" + "=" * 60)
        print(f"{emoji} {title}")
        print("=" * 60)
    return


@app.cell
def _():
    #==============================================
    #1. 데이터 로드 및 전처리
    #==============================================

    return


if __name__ == "__main__":
    app.run()
