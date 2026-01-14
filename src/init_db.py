import pandas as pd
import sqlite3
import os

def create_db():
    db_path = 'data/processed/kleague.db'
    if os.path.exists(db_path):
        print(f"'{db_path}' exists. Overwriting...")
    
    print("CSV 데이터 로딩 중 (data/raw/raw_data.csv, data/raw/match_info.csv)...")
    try:
        raw_data = pd.read_csv('data/raw/raw_data.csv')
        match_info = pd.read_csv('data/raw/match_info.csv')
        
        # SQLite 연결
        conn = sqlite3.connect(db_path)
        
        print("테이블 생성 중...")
        raw_data.to_sql('raw_data', conn, if_exists='replace', index=False)
        match_info.to_sql('match_info', conn, if_exists='replace', index=False)
        
        # 인덱스 생성 (성능 향상)
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX idx_raw_game_id ON raw_data(game_id)")
        cursor.execute("CREATE INDEX idx_match_game_id ON match_info(game_id)")
        
        conn.close()
        print(f"완료! '{db_path}'가 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    create_db()
