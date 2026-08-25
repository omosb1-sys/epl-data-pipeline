import os

import mysql.connector

def update_manutd_tactics():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=os.getenv("MYSQL_PASSWORD", ""),
            database="epl_x_db",
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()

        # 대런 플레처 (임시) 전술 정보 업데이트
        # 플레처는 더욱 전통적인 4-3-3 또는 4-2-3-1을 선호하며, 아모림의 3-4-3을 폐기하고 안정성을 추구 중.
        
        new_formation = "4-3-3 (Interim)"
        new_tactics = "안정적인 중원 장악과 빠른 측면 전개. 아모림의 3백을 버리고 4백으로 회귀하여 수비 안정화에 집중 (플레처 임시 체제)"

        print(f"🔄 맨유 전술 정보를 업데이트합니다: {new_formation}")
        
        # 전술 및 설명 업데이트
        cursor.execute("""
            UPDATE clubs 
            SET tactics_formation = %s, 
                tactics_desc = %s 
            WHERE team_name = '맨체스터 유나이티드'
        """, (new_formation, new_tactics))

        conn.commit()
        print("✅ 업데이트 완료.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    update_manutd_tactics()
