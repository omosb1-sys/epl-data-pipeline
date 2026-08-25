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

def absolute_final_img():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # [플랜 D] Placehold.co 사용 (이미지 생성 서비스)
    # 절대 깨질 수 없는 심플한 이미지
    concrete_url = "https://placehold.co/600x400/png?text=Old+Trafford"
    
    sql = """
        UPDATE clubs 
        SET stadium_img = %s 
        WHERE team_name LIKE '%맨체스터%유나이티드%'
    """
    
    try:
        cursor.execute(sql, (concrete_url,))
        conn.commit()
        print(f"✅ 절대 방어 이미지 설정 완료! (Placehold.co 사용)")
        print(f"🔗 새 URL: {concrete_url}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    absolute_final_img()
