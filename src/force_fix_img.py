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

def force_update_manutd_img_safe():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # [핵심] 특수문자호환성이 좋은 '썸네일 버전' URL로 교체
    # 원본보다 로딩도 빠르고 에러가 없습니다.
    safe_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Old_Trafford%2C_Manchester.jpg/800px-Old_Trafford%2C_Manchester.jpg"
    
    # 맨유 찾아서 업데이트
    sql = """
        UPDATE clubs 
        SET stadium_img = %s 
        WHERE team_name LIKE '%맨체스터%유나이티드%'
    """
    
    try:
        cursor.execute(sql, (safe_url,))
        conn.commit()
        
        print(f"✅ [완료] 맨유 경기장 이미지를 '안전한 썸네일 버전'으로 교체했습니다!")
        print(f"🔗 적용된 URL: {safe_url}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    force_update_manutd_img_safe()
