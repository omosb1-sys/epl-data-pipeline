import mysql.connector
from datetime import datetime

def connect_to_db():
    """
    MySQL 데이터베이스에 연결하는 함수입니다.
    Mac 로컬 환경의 설정을 사용합니다.
    """
    try:
        # 주의: password 부분은 막내님(본인)이 설정한 비번으로 꼭 바꿔주세요!
        connection = mysql.connector.connect(
            host="localhost",
            user="root",         # 보통 mac mysql 기본 유저는 root입니다. 아니면 변경하세요.
            password="0623os1978", # <--- 여기에 워크벤치 들어갈 때 쓰는 비번 입력!
            database="epl_x_db"
        )
        
        if connection.is_connected():
            print("✅ MySQL 데이터베이스 연결 성공!")
            return connection
            
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return None

def insert_clean_sheet_data(connection):
    """
    맨유의 데이터(샘플)를 DB에 넣는 함수
    """
    cursor = connection.cursor()
    
    # 1. 아까 만든 clubs 테이블에 데이터 넣기
    # "이미 있으면 무시하고(IGNORE), 없으면 넣어라"하는 안전한 쿼리
    sql_club = """
    INSERT IGNORE INTO clubs (team_name, manager_name, power_index) 
    VALUES (%s, %s, %s)
    """
    val_club = ("Man Utd", "Ruben Amorim", 85)
    
    cursor.execute(sql_club, val_club)
    
    # 2. 이번 주 경기 일정 넣기 (맨유 vs 아스날, 가상)
    sql_match = """
    INSERT INTO matches (home_team, away_team, kick_off_time, weather)
    VALUES (%s, %s, %s, %s)
    """
    # 2024년 1월 1일 오후 8시 경기, 비 옴
    val_match = ("Man Utd", "Arsenal", datetime(2025, 1, 15, 20, 0, 0), "Rainy")
    
    cursor.execute(sql_match, val_match)
    
    # 변경사항 저장 (Commit)
    connection.commit()
    print(f"✅ 데이터 입력 완료: {cursor.rowcount}개 행이 추가되었습니다.")

def fetch_data(connection):
    """
    DB에 잘 들어갔는지 확인하는 함수
    """
    cursor = connection.cursor()
    
    print("\n[현재 DB 저장된 클럽 목록]")
    cursor.execute("SELECT * FROM clubs")
    for row in cursor.fetchall():
        print(row)
        
    print("\n[경기 일정 목록]")
    cursor.execute("SELECT * FROM matches")
    for row in cursor.fetchall():
        print(row)

# --- 메인 실행 ---
if __name__ == "__main__":
    conn = connect_to_db()
    if conn:
        insert_clean_sheet_data(conn)
        fetch_data(conn)
        conn.close()
