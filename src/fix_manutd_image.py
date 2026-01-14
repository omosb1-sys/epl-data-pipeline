import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def fix_manutd_img():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 올드 트래포드 이미지 (위키미디어)
    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Old_Trafford%2C_Manchester.jpg/600px-Old_Trafford%2C_Manchester.jpg"
    
    # 쿼리: 맨유 팀을 찾아서 stadium_img 컬럼을 강제로 업데이트
    sql = "UPDATE clubs SET stadium_img = %s WHERE team_name = '맨체스터 유나이티드'"
    
    try:
        cursor.execute(sql, (img_url,))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"✅ 수정 완료! 맨체스터 유나이티드의 경기장 이미지가 복구되었습니다.")
            print(f"👉 적용된 URL: {img_url}")
        else:
            print("⚠️ 수정 실패: '맨체스터 유나이티드'라는 이름의 팀을 DB에서 찾을 수 없습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    fix_manutd_img()
