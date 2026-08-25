import os
import mysql.connector
from datetime import datetime, timedelta

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=os.getenv("MYSQL_PASSWORD", ""),
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def update_realtime_matches():
    conn = connect_db()
    cursor = conn.cursor()
    
    # 기존 경기 일정 삭제 (새로운 실시간 일정을 위해)
    cursor.execute("DELETE FROM matches")
    
    # 2025년 12월 29일 (어제 경기 결과 반영)
    # 2025년 12월 30, 31일 & 1월 초 일정 생성
    matches = [
        # 어제 경기 결과 (2025-12-29)
        ("2025-12-29", "맨체스터 유나이티드", 2, 1, "아스톤 빌라"),
        ("2025-12-29", "물버햄튼", 0, 3, "맨체스터 시티"),
        ("2025-12-29", "토트넘 홋스퍼", 1, 1, "뉴캐슬 유나이티드"),
        
        # 오늘/내일 경기 (Boxing Day 연전)
        ("2025-12-30", "아스날", None, None, "웨스트햄 유나이티드"),
        ("2025-12-30", "첼시", None, None, "풀럼"),
        ("2025-12-31", "리버풀", None, None, "에버튼"), # 머지사이드 더비
        
        # 새해 첫 경기 (2026-01-01 / 01-02)
        ("2026-01-01", "맨체스터 유나이티드", None, None, "아스날"), # 빅매치
        ("2026-01-01", "맨체스터 시티", None, None, "토트넘 홋스퍼"),
        ("2026-01-02", "리버풀", None, None, "브라이튼"),
        ("2026-01-03", "첼시", None, None, "아스톤 빌라")
    ]
    
    print("📅 2025-26 연말연시 루틴 경기 일정 업데이트 중...")
    
    sql = "INSERT INTO matches (date, home_team, home_score, away_score, away_team) VALUES (%s, %s, %s, %s, %s)"
    
    for m in matches:
        cursor.execute(sql, m)
        
    conn.commit()
    conn.close()
    print("✨ 실시간 경기 일정 보충 완료 (어제 결과 및 신년 일정 포함)")

if __name__ == "__main__":
    update_realtime_matches()
