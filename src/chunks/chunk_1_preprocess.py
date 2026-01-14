"""
Chunk 1: Data Preprocessing for K-League Win Rate Analysis
==========================================================
이 모듈은 match_info.csv를 로드하여 팀 기준의 승/무/패 데이터를 생성합니다.

작성자: Antigravity (Gemini 3)
작성일: 2026-01-14
"""

import pandas as pd
import os
import sys

# 프로젝트 루트 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data/raw/match_info.csv")

def load_and_preprocess_data() -> pd.DataFrame:
    """
    데이터를 로드하고 팀 기준으로 재구조화하여 승패 결과를 계산합니다.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_PATH}")
        
    print(f"📂 데이터 로드 중: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # 필요한 컬럼만 선택
    cols = ['season_id', 'game_id', 'home_team_name', 'away_team_name', 'home_score', 'away_score']
    df = df[cols].copy()
    
    # 홈 팀 기준 데이터
    home_df = df.rename(columns={
        'home_team_name': 'team', 
        'away_team_name': 'opponent',
        'home_score': 'goals_for',
        'away_score': 'goals_against'
    })
    home_df['is_home'] = True
    
    # 원정 팀 기준 데이터
    away_df = df.rename(columns={
        'away_team_name': 'team', 
        'home_team_name': 'opponent',
        'away_score': 'goals_for',
        'home_score': 'goals_against'
    })
    away_df['is_home'] = False
    
    # 데이터 병합 (모든 경기를 팀 관점으로 펼침)
    long_df = pd.concat([home_df, away_df], ignore_index=True)
    
    # 승/무/패 판별 로직
    # 승: 득점 > 실점
    # 무: 득점 == 실점
    # 패: 득점 < 실점
    conditions = [
        (long_df['goals_for'] > long_df['goals_against']),
        (long_df['goals_for'] == long_df['goals_against']),
        (long_df['goals_for'] < long_df['goals_against'])
    ]
    choices = ['Win', 'Draw', 'Loss']
    long_df['result'] = np.select(conditions, choices, default='Unknown')
    
    # 승점 계산 (승 3점, 무 1점)
    long_df['points'] = long_df['result'].map({'Win': 3, 'Draw': 1, 'Loss': 0})
    
    print(f"✅ 전처리 완료: 총 {len(long_df)}개의 팀별 경기 기록 생성됨.")
    return long_df

if __name__ == "__main__":
    import numpy as np # 내부 사용을 위해 여기서 import
    try:
        df_processed = load_and_preprocess_data()
        print(df_processed.head())
        
        # 중간 결과 저장 (선택 사항)
        save_path = os.path.join(BASE_DIR, "data/processed/team_match_results.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_processed.to_csv(save_path, index=False)
        print(f"💾 중간 데이터 저장 완료: {save_path}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
