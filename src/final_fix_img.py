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

def final_rescue_img():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # [플랜 C] Unsplash의 아주 튼튼한 '축구장' 이미지 (절대 안 깨짐)
    # 붉은색 계열 경기장 느낌으로 선택
    robust_url = "https://images.unsplash.com/photo-1522778119026-d647f0565c6a?auto=format&fit=crop&w=800&q=80"
    
    sql = """
        UPDATE clubs 
        SET stadium_img = %s 
        WHERE team_name LIKE '%맨체스터%유나이티드%'
    """
    
    try:
        cursor.execute(sql, (robust_url,))
        conn.commit()
        print(f"✅ 구원 투수 등판 완료! (Unsplash 이미지 사용)")
        print(f"🔗 새 URL: {robust_url}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    final_rescue_img()
