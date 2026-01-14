import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def check_teams():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT team_name, stadium_img FROM clubs")
    rows = cursor.fetchall()
    
    print("📋 현재 DB에 저장된 구단 목록:")
    print("-" * 40)
    for row in rows:
        team_name = row[0]
        has_img = "✅이미지있음" if row[1] else "❌이미지없음"
        print(f"• {team_name} ({has_img})")
    
    conn.close()

if __name__ == "__main__":
    check_teams()
