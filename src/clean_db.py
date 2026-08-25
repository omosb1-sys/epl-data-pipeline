import os
import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=os.getenv("MYSQL_PASSWORD", ""),
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def clean_and_korean_patch():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    print("🧹 데이터 대청소 및 한글 패치 시작...")

    # 1. 기존 데이터 싹 비우기 (중복 제거를 위해)
    # 걱정 마세요, 바로 다시 채울 거니까요!
    cursor.execute("DELETE FROM clubs") 
    cursor.execute("DELETE FROM matches")
    print("🗑️ 기존 지저분한 데이터 삭제 완료")
    
    # 2. 한글 이름으로 깔끔하게 다시 넣기
    korean_clubs = [
        ("맨체스터 유나이티드", "루벤 아모림", 85),
        ("맨체스터 시티", "펩 과르디올라", 96),
        ("아스날", "미켈 아르테타", 92),
        ("리버풀", "아르네 슬롯", 94),
        ("첼시", "엔조 마레스카", 88),
        ("토트넘 홋스퍼", "안지 포스테코글루", 86),
        ("뉴캐슬 유나이티드", "에디 하우", 83),
        ("아스톤 빌라", "우나이 에메리", 84),
        ("울버햄튼", "게리 오닐", 74),
        ("브라이튼", "파비안 휘르첼러", 80)
    ]
    
    sql_club = "INSERT INTO clubs (team_name, manager_name, power_index) VALUES (%s, %s, %s)"
    cursor.executemany(sql_club, korean_clubs)
    print("✅ 한글 팀명 업데이트 완료!")

    # 3. 경기 일정도 한글로 다시 넣기
    korean_matches = [
        ("맨체스터 유나이티드", "아스날", "2025-01-15 20:00:00", "비(Rainy)"),
        ("리버풀", "맨체스터 시티", "2025-01-18 12:30:00", "맑음(Sunny)"),
        ("토트넘 홋스퍼", "첼시", "2025-01-20 19:45:00", "흐림(Cloudy)")
    ]
    sql_match = "INSERT INTO matches (home_team, away_team, kick_off_time, weather) VALUES (%s, %s, %s, %s)"
    cursor.executemany(sql_match, korean_matches)
    
    conn.commit()
    conn.close()
    print("✨ DB 정비 작업 끝! 앱을 새로고침 하세요.")

if __name__ == "__main__":
    clean_and_korean_patch()
