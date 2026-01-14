import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def populate_all_teams():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 2024-25 시즌 프리미어리그 나머지 구단 데이터
    # (팀명, 감독(임시/현재), 예상 전력)
    rest_epl_teams = [
        ("Tottenham", "Ange Postecoglou", 86),
        ("Aston Villa", "Unai Emery", 84),
        ("Newcastle", "Eddie Howe", 83),
        ("Brighton", "Fabian Hurzeler", 80),
        ("West Ham", "Julen Lopetegui", 78),
        ("Crystal Palace", "Oliver Glasner", 76),
        ("Fulham", "Marco Silva", 75),
        ("Bournemouth", "Andoni Iraola", 75),
        ("Wolves", "Gary O'Neil", 74),
        ("Everton", "Sean Dyche", 73),
        ("Brentford", "Thomas Frank", 74),
        ("Nottm Forest", "Nuno Espirito Santo", 72),
        ("Leicester", "Steve Cooper", 70),
        ("Southampton", "Russell Martin", 69),
        ("Ipswich", "Kieran McKenna", 68)
    ]
    
    sql = "INSERT IGNORE INTO clubs (team_name, manager_name, power_index) VALUES (%s, %s, %s)"
    
    try:
        print("⏳ EPL 전체 구단 데이터 업데이트 중...")
        cursor.executemany(sql, rest_epl_teams)
        conn.commit()
        print(f"✅ 구단 데이터 추가 완료! 총 {cursor.rowcount}개 팀이 더해졌습니다.")
        
    except mysql.connector.Error as err:
        print(f"❌ 오류 발생: {err}")
    finally:
        conn.close()

if __name__ == "__main__":
    populate_all_teams()
