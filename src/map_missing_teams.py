import mysql.connector
import os

def map_missing():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )
    cursor = conn.cursor()
    
    # 누락된 팀들 수동 매핑
    targets = {
        "에버": "stadiums/everton.png",    # 에버턴/에버튼
        "리버풀": "stadiums/liverpool.jpg", # 리버풀
        "맨체스터 시티": "stadiums/man_city.jpg", # 이름에 포함
        "사우": "stadiums/s_hampton.png"    # 사우샘프턴/사우스햄튼
    }
    
    print("🚑 누락된 팀 연결 시도...")
    
    for team_prefix, path in targets.items():
        if os.path.exists(path):
            sql = "UPDATE clubs SET stadium_img = %s WHERE team_name LIKE %s"
            cursor.execute(sql, (path, f"%{team_prefix}%"))
            print(f"✅ '{team_prefix}' 포함 팀 -> {path} ({cursor.rowcount}행 수정)")
        else:
            print(f"❌ 파일 없음: {path}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    map_missing()
