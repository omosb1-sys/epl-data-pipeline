import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def restore_full_squad():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 아까 누락된 나머지 10개 팀 (한글 패치)
    missing_teams = [
        ("크리스탈 팰리스", "올리버 글라스너", 76),
        ("풀럼", "마르코 실바", 75),
        ("본머스", "안도니 이라올라", 75),
        ("웨스트햄 유나이티드", "훌렌 로페테기", 78),
        ("에버튼", "션 다이치", 73),
        ("브렌트포드", "토마스 프랭크", 74),
        ("노팅엄 포레스트", "누누 산투", 72),
        ("레스터 시티", "스티브 쿠퍼", 70),
        ("사우스햄튼", "러셀 마틴", 69),
        ("입스위치 타운", "키어런 맥케나", 68)
    ]
    
    sql = "INSERT INTO clubs (team_name, manager_name, power_index) VALUES (%s, %s, %s)"
    
    try:
        cursor.executemany(sql, missing_teams)
        conn.commit()
        print(f"✅ 나머지 {cursor.rowcount}개 팀 복구 완료! 이제 총 20개 팀입니다.")
    except Exception as e:
        print(f"⚠️ 복구 중 오류: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    restore_full_squad()
