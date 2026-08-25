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

def bypass_wiki_server():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # [플랜 B] 위키미디어 대신 외부 이미지 사용 (맨유 공식 홈페이지나 뉴스 사이트 소스)
    # Goal.com 이미지 소스 (안정적임)
    new_url = "https://assets.goal.com/v3/assets/bltcc7a7ffd2fbf71f5/blt5e33620703f5724d/60db7138b251210f6793e25d/b86aef151743a429007425176043126f555d720b.jpg"
    
    sql = """
        UPDATE clubs 
        SET stadium_img = %s 
        WHERE team_name LIKE '%맨체스터%유나이티드%'
    """
    
    try:
        cursor.execute(sql, (new_url,))
        conn.commit()
        print(f"✅ 위키미디어 서버 오류 우회 완료! (Goal.com 이미지 사용)")
        print(f"🔗 새 URL: {new_url}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    bypass_wiki_server()
