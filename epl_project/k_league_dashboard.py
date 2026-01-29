"""
K-리그 데이터 분석 대시보드
===========================
Streamlit 기반 인터랙티브 웹 앱

Author: Antigravity AI
Date: 2026-01-22
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="K-리그 데이터 분석",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

BASE_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data"


@st.cache_data
def load_data():
    """데이터 로드 및 전처리"""
    match_info = pd.read_csv(f"{BASE_PATH}/match_info.csv")
    raw_data = pd.read_csv(f"{BASE_PATH}/raw_data.csv")
    
    game_stats = raw_data.groupby(['game_id', 'team_id']).agg(
        total_actions=('action_id', 'count'),
        total_passes=('type_name', lambda x: (x == 'Pass').sum()),
        total_shots=('type_name', lambda x: (x == 'Shot').sum()),
        successful_actions=('result_name', lambda x: (x == 'Successful').sum()),
        avg_x_position=('start_x', 'mean'),
        unique_players=('player_id', 'nunique')
    ).reset_index()
    
    game_stats['success_rate'] = game_stats['successful_actions'] / game_stats['total_actions']
    game_stats['pass_ratio'] = game_stats['total_passes'] / game_stats['total_actions']
    
    df = game_stats.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score']],
        on='game_id', how='left'
    )
    
    def get_result(row):
        if row['team_id'] == row['home_team_id']:
            return 'Win' if row['home_score'] > row['away_score'] else ('Lose' if row['home_score'] < row['away_score'] else 'Draw')
        else:
            return 'Win' if row['away_score'] > row['home_score'] else ('Lose' if row['away_score'] < row['home_score'] else 'Draw')
    
    df['result'] = df.apply(get_result, axis=1)
    
    return df, match_info, raw_data


def main():
    """메인 대시보드"""
    
    # 헤더
    st.markdown('<div class="main-header">⚽ K-리그 데이터 분석 대시보드</div>', unsafe_allow_html=True)
    st.markdown("**Powered by Antigravity AI** | 실시간 인터랙티브 분석")
    
    # 데이터 로드
    with st.spinner('데이터 로딩 중...'):
        df, match_info, raw_data = load_data()
    
    # 사이드바
    st.sidebar.title("📊 분석 옵션")
    st.sidebar.markdown("---")
    
    analysis_type = st.sidebar.selectbox(
        "분석 유형 선택",
        ["📈 대시보드 개요", "🔍 탐색적 분석 (EDA)", "🤖 머신러닝 예측", "🎨 3D 시각화"]
    )
    
    # 탭 구성
    if analysis_type == "📈 대시보드 개요":
        show_dashboard(df, match_info)
    elif analysis_type == "🔍 탐색적 분석 (EDA)":
        show_eda(df)
    elif analysis_type == "🤖 머신러닝 예측":
        show_ml(df)
    elif analysis_type == "🎨 3D 시각화":
        show_3d_viz(df)


def show_dashboard(df, match_info):
    """대시보드 개요"""
    st.header("📊 대시보드 개요")
    
    # KPI 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 경기 수", f"{match_info['game_id'].nunique()}경기")
    with col2:
        st.metric("평균 액션 수", f"{df['total_actions'].mean():.0f}")
    with col3:
        st.metric("평균 성공률", f"{df['success_rate'].mean():.1%}")
    with col4:
        st.metric("평균 슈팅 수", f"{df['total_shots'].mean():.1f}")
    
    st.markdown("---")
    
    # 승/무/패 분포
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("승/무/패 분포")
        result_counts = df['result'].value_counts()
        fig = px.pie(
            values=result_counts.values,
            names=result_counts.index,
            color=result_counts.index,
            color_discrete_map={'Win': 'green', 'Draw': 'orange', 'Lose': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("슈팅 수 분포")
        fig = px.histogram(df, x='total_shots', nbins=30, color='result',
                          color_discrete_map={'Win': 'green', 'Draw': 'orange', 'Lose': 'red'})
        st.plotly_chart(fig, use_container_width=True)


def show_eda(df):
    """탐색적 데이터 분석"""
    st.header("🔍 탐색적 데이터 분석 (EDA)")
    
    # 변수 선택
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("X축 변수", ['total_shots', 'total_passes', 'success_rate', 'avg_x_position'])
    with col2:
        y_var = st.selectbox("Y축 변수", ['success_rate', 'total_actions', 'total_shots', 'pass_ratio'])
    
    # 산점도
    fig = px.scatter(
        df, x=x_var, y=y_var, color='result',
        size='total_actions',
        hover_data=['total_shots', 'success_rate'],
        color_discrete_map={'Win': 'green', 'Draw': 'orange', 'Lose': 'red'},
        title=f'{x_var} vs {y_var}'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 상관관계 히트맵
    st.subheader("상관관계 행렬")
    corr_cols = ['total_actions', 'total_passes', 'total_shots', 'success_rate']
    corr_matrix = df[corr_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)


def show_ml(df):
    """머신러닝 예측"""
    st.header("🤖 머신러닝 예측")
    
    st.info("RandomForest 모델을 사용하여 승/무/패를 예측합니다.")
    
    if st.button("🚀 모델 학습 시작"):
        with st.spinner('모델 학습 중...'):
            # 데이터 준비
            feature_cols = ['total_actions', 'total_passes', 'total_shots', 
                           'success_rate', 'pass_ratio', 'avg_x_position', 'unique_players']
            
            X = df[feature_cols].fillna(0)
            y = df['result']
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # 결과 표시
            st.success(f"✅ 모델 학습 완료! 정확도: {accuracy:.1%}")
            
            # Feature Importance
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                        title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)


def show_3d_viz(df):
    """3D 시각화"""
    st.header("🎨 3D 인터랙티브 시각화")
    
    st.subheader("3D 산점도: 슈팅 vs 성공률 vs 액션")
    
    color_map = {'Win': 'green', 'Draw': 'orange', 'Lose': 'red'}
    
    fig = go.Figure()
    
    for result in ['Win', 'Draw', 'Lose']:
        data = df[df['result'] == result]
        fig.add_trace(go.Scatter3d(
            x=data['total_shots'],
            y=data['success_rate'],
            z=data['total_actions'],
            mode='markers',
            name=result,
            marker=dict(
                size=6,
                color=color_map[result],
                opacity=0.7
            )
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='슈팅 수',
            yaxis_title='성공률',
            zaxis_title='총 액션'
        ),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
