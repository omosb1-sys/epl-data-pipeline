try:
    import shap
except ImportError:
    shap = None
import pandas as pd
import numpy as np
import os
import streamlit as st
import altair as alt

def get_real_shap_analysis(model, input_scaled, feature_names):
    """
    실제 SHAP 라이브러리를 사용하여 입력 데이터의 기여도를 분석합니다.
    """
    if shap is None:
        st.warning("⚠️ SHAP 라이브러리가 설치되지 않아 엔진 최적화 대기 중입니다. (pip install shap required)")
        return None
    try:
        # TreeExplainer는 RandomForest 등의 트리 기반 모델에 최적화되어 있습니다.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        
        # Binary/Multi-class Classification 대응
        # shap_values의 형태는 모델과 라이브러리 버전에 따라 다를 수 있음
        if isinstance(shap_values, list): 
            # 리스트 형태 (주로 sklearn Binary Classification)
            # 승리(Win=2)에 대한 기여도를 보려면 3개의 클래스 중 마지막 인덱스 사용 시도
            # 하지만 안전하게 첫 번째 샘플의 기여도만 추출
            item_shap = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3: 
            # [샘플, 피처, 클래스] (최신 SHAP + Multi-class)
            # 클래스 인덱스 2 (Win) 에 대한 기여도 추출
            item_shap = shap_values[0, :, 2] if shap_values.shape[2] > 2 else shap_values[0, :, 1]
        else:
            item_shap = shap_values[0]

        # 데이터프레임 변환
        df_shap = pd.DataFrame({
            'Feature': feature_names,
            'Impact': item_shap
        })
        
        # 한글 이름 매핑 (UI용)
        kor_map = {
            # Legacy Features
            'recent_goals': '최근 득점력',
            'goals_scored': '최근 득점력',
            'goals_conceded': '수비 안정도',
            'elo': '객관적 전력(ELO)',
            'form': '최근 경기 흐름',
            # ADX Features
            'ADX': '추세 강도(ADX)',
            '+DI': '상승 에너지(+DI)',
            '-DI': '하락 에너지(-DI)',
            # Common
            'home_advantage': '홈 어드밴티지',
            'rest_days': '에너지 레벨(휴식)',
            'injury_level': '스쿼드 가용성'
        }
        df_shap['Feature'] = df_shap['Feature'].map(lambda x: kor_map.get(x, x))
        
        return df_shap
    except Exception as e:
        import traceback
        st.error(f"SHAP 연산 중 오류: {e}")
        st.caption(f"DEBUG: {traceback.format_exc()}")
        return None


def display_shap_chart(df_shap):
    """
    분석된 SHAP 데이터를 Altair 차트로 시각화합니다.
    """
    if df_shap is None or not isinstance(df_shap, pd.DataFrame):
        st.info("💡 AI 분석 기여도 데이터를 생성하는 중입니다...")
        return
        
    # [Fix] 데이터 타입 판별 오류 방지를 위한 정밀 정제 및 복사
    df_chart = df_shap.copy()
    df_chart['Impact'] = pd.to_numeric(df_chart['Impact'], errors='coerce').fillna(0).astype(float)
    
    # [KOR Translation] 분석 변수명 한글화
    name_map = {
        'ADX': '추세 강도 (ADX)',
        '-DI': '하락/수비 압박 (-DI)',
        'home_advantage': '홈 경기 이점',
        'rest_days': '선수단 휴식일',
        '+DI': '상승/공격 지지 (+DI)',
        'Impact': '승리 기여도'
    }
    if 'Feature' in df_chart.columns:
        df_chart['Feature'] = df_chart['Feature'].replace(name_map)
        
    chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X('Impact:Q', title='승리 기여도 (AI 영향력)'),
        y=alt.Y('Feature:N', sort='-x', title='주요 분석 지표'),
        # [Fix] 수치(datum) 기반 조건부 색상 결정
        color=alt.condition(
            'datum.Impact > 0',
            alt.value('#4CAF50'), # 상승 (Green)
            alt.value('#E91E63')  # 하락 (Pink/Red)
        ),
        tooltip=[
            alt.Tooltip('Feature:N', title='지표'),
            alt.Tooltip('Impact:Q', format='.2f', title='기여도')
        ]
    ).properties(
        height=300,
        title='🛡️ AI 판단 근거 상세 분석'
    )
    
    st.altair_chart(chart, width="stretch")
    st.caption("※ **실시간 AI 분석**: 머신러닝 모델이 각 지표를 분석하여 도출한 '승리 확률에 미친 영향'입니다.")
