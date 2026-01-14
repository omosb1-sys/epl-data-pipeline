import mysql.connector
from datetime import datetime, timedelta

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978", # 막내님 비번
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def populate_data():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 1. 빅5 클럽 정보 (아모림, 펩, 슬롯 등 최신 감독 반영)
    clubs_data = [
        ("Man City", "Pep Guardiola", 96),
        ("Liverpool", "Arne Slot", 94),
        ("Arsenal", "Mikel Arteta", 92),
        ("Chelsea", "Enzo Maresca", 88),
        # 맨유는 이미 들어있지만 중복 방지(IGNORE)로 안전하게
        ("Man Utd", "Ruben Amorim", 85)
    ]
    
    sql_club = "INSERT IGNORE INTO clubs (team_name, manager_name, power_index) VALUES (%s, %s, %s)"
    
    # 2. 이번 주 주요 빅매치 일정 (가상 데이터)
    matches_data = [
        ("Liverpool", "Man City", datetime(2025, 1, 18, 12, 30), "Sunny"), # 낮 경기 (리버풀 홈)
        ("Arsenal", "Chelsea", datetime(2025, 1, 19, 20, 0), "Cloudy"),    # 저녁 경기
        ("Man Utd", "Liverpool", datetime(2025, 2, 1, 15, 0), "Rainy"),    # 라이벌전
        ("Chelsea", "Man City", datetime(2025, 2, 5, 20, 45), "Rainy")
    ]
    
    sql_match = "INSERT INTO matches (home_team, away_team, kick_off_time, weather) VALUES (%s, %s, %s, %s)"

    try:
        # 클럽 데이터 입력
        print("⏳ 빅5 클럽 데이터 입력 중...")
        # executemany를 쓰면 데이터를 한방에 넣을 수 있습니다 (시니어의 꿀팁)
        cursor.executemany(sql_club, clubs_data)
        
        # 매치 데이터 입력
        print("⏳ 경기 일정 생성 중...")
        cursor.executemany(sql_match, matches_data)
        
        conn.commit()
        print(f"✅ 성공! 클럽 {cursor.rowcount}개 및 경기 일정이 추가되었습니다.")
        
    except mysql.connector.Error as err:
        print(f"❌ 데이터 입력 실패: {err}")
    finally:
        conn.close()

if __name__ == "__main__":
    populate_data()
