"""
K-리그 3D 인터랙티브 시각화
============================
Plotly를 사용한 3D 산점도, 히트맵, 애니메이션

Author: Antigravity AI
Date: 2026-01-22
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

BASE_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data"
OUTPUT_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/output"


def load_data():
    """데이터 로드 및 전처리"""
    print("📂 데이터 로드 중...")
    
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
    
    print(f"✅ 데이터 로드 완료: {df.shape}")
    return df


def create_3d_scatter(df):
    """3D 산점도: 슈팅 vs 성공률 vs 승패"""
    print("\n🎨 3D 산점도 생성 중...")
    
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
                size=8,
                color=color_map[result],
                opacity=0.7,
                line=dict(color='white', width=0.5)
            ),
            text=[f"슈팅: {s}<br>성공률: {sr:.2f}<br>액션: {a}" 
                  for s, sr, a in zip(data['total_shots'], data['success_rate'], data['total_actions'])],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='K-리그 경기 분석: 3D 공간에서의 승/무/패 패턴',
            font=dict(size=20, family='AppleGothic')
        ),
        scene=dict(
            xaxis_title='총 슈팅 수',
            yaxis_title='성공률',
            zaxis_title='총 액션 수',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1200,
        height=800,
        template='plotly_dark'
    )
    
    output_file = f"{OUTPUT_PATH}/3d_scatter_interactive.html"
    fig.write_html(output_file)
    print(f"✅ 저장: {output_file}")
    
    return fig


def create_3d_surface(df):
    """3D 표면 그래프: 슈팅 수 x 성공률 → 승률"""
    print("\n🎨 3D 표면 그래프 생성 중...")
    
    # 그리드 생성
    shots_bins = np.linspace(df['total_shots'].min(), df['total_shots'].max(), 20)
    success_bins = np.linspace(df['success_rate'].min(), df['success_rate'].max(), 20)
    
    df['shot_bin'] = pd.cut(df['total_shots'], bins=shots_bins, labels=False)
    df['success_bin'] = pd.cut(df['success_rate'], bins=success_bins, labels=False)
    
    win_rate_grid = df.groupby(['shot_bin', 'success_bin'])['result'].apply(
        lambda x: (x == 'Win').sum() / len(x) if len(x) > 0 else 0
    ).unstack(fill_value=0)
    
    fig = go.Figure(data=[go.Surface(
        z=win_rate_grid.values,
        x=shots_bins[:-1],
        y=success_bins[:-1],
        colorscale='Viridis',
        colorbar=dict(title='승률')
    )])
    
    fig.update_layout(
        title='슈팅 수 & 성공률에 따른 승률 분포 (3D 표면)',
        scene=dict(
            xaxis_title='슈팅 수',
            yaxis_title='성공률',
            zaxis_title='승률'
        ),
        width=1200,
        height=800
    )
    
    output_file = f"{OUTPUT_PATH}/3d_surface_win_rate.html"
    fig.write_html(output_file)
    print(f"✅ 저장: {output_file}")
    
    return fig


def create_animated_timeline(df):
    """애니메이션: 경기 흐름 시각화"""
    print("\n🎨 애니메이션 타임라인 생성 중...")
    
    # 경기 ID별로 정렬
    df_sorted = df.sort_values('game_id').reset_index(drop=True)
    df_sorted['game_index'] = df_sorted.index
    
    fig = px.scatter(
        df_sorted,
        x='total_shots',
        y='success_rate',
        animation_frame='game_index',
        animation_group='team_id',
        size='total_actions',
        color='result',
        hover_name='game_id',
        color_discrete_map={'Win': 'green', 'Draw': 'orange', 'Lose': 'red'},
        range_x=[0, df['total_shots'].max() * 1.1],
        range_y=[0, df['success_rate'].max() * 1.1],
        title='K-리그 경기 흐름 애니메이션'
    )
    
    fig.update_layout(
        width=1200,
        height=700,
        xaxis_title='슈팅 수',
        yaxis_title='성공률'
    )
    
    output_file = f"{OUTPUT_PATH}/animated_timeline.html"
    fig.write_html(output_file)
    print(f"✅ 저장: {output_file}")
    
    return fig


def create_correlation_3d(df):
    """3D 상관관계 네트워크"""
    print("\n🎨 3D 상관관계 네트워크 생성 중...")
    
    corr_cols = ['total_actions', 'total_passes', 'total_shots', 'success_rate']
    corr_matrix = df[corr_cols].corr()
    
    # 3D 히트맵
    fig = go.Figure(data=[go.Surface(
        z=corr_matrix.values,
        x=corr_cols,
        y=corr_cols,
        colorscale='RdBu',
        cmid=0,
        colorbar=dict(title='상관계수')
    )])
    
    fig.update_layout(
        title='변수 간 상관관계 3D 히트맵',
        scene=dict(
            xaxis_title='변수 1',
            yaxis_title='변수 2',
            zaxis_title='상관계수'
        ),
        width=1000,
        height=800
    )
    
    output_file = f"{OUTPUT_PATH}/3d_correlation_heatmap.html"
    fig.write_html(output_file)
    print(f"✅ 저장: {output_file}")
    
    return fig


def main():
    """메인 실행"""
    print("\n" + "🎨" * 25)
    print("   K-리그 3D 인터랙티브 시각화")
    print("🎨" * 25 + "\n")
    
    df = load_data()
    
    create_3d_scatter(df)
    create_3d_surface(df)
    create_animated_timeline(df)
    create_correlation_3d(df)
    
    print("\n" + "✅" * 25)
    print("   모든 3D 시각화 완료!")
    print("   브라우저에서 HTML 파일을 열어보세요!")
    print("✅" * 25)


if __name__ == "__main__":
    main()
