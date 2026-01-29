
import mysql.connector

def update_manutd_manager():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="0623os1978",
            database="epl_x_db",
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()

        # 맨유 감독 업데이트: 후벤 아모림 경질 -> 대런 플레처(임시)
        # 2026년 1월 5일 경질, 현재 플레처 임시 체제. 차기 임시 감독으로 솔샤르/캐릭 거론 중.
        
        new_manager = "대런 플레처 (임시, Darren Fletcher)"
        news_update = "후벤 아모림 경질(1/5), 대런 플레처 임시 감독 선임. 솔샤르/캐릭 복귀설 파다함."

        print(f"🔄 맨유 감독 정보를 업데이트합니다: {new_manager}")
        
        # 1. 감독 이름 변경
        cursor.execute("UPDATE clubs SET manager_name = %s WHERE team_name = '맨체스터 유나이티드'", (new_manager,))
        
        # 2. 뉴스/이적 루머 란에 감독 경질 소식 추가
        cursor.execute("UPDATE clubs SET winter_rumors_out = CONCAT(winter_rumors_out, ', ', %s) WHERE team_name = '맨체스터 유나이티드'", ("아모림 감독 경질",))
        cursor.execute("UPDATE clubs SET winter_rumors_in = CONCAT(winter_rumors_in, ', ', %s) WHERE team_name = '맨체스터 유나이티드'", ("솔샤르/캐릭 감독 복귀설",))

        conn.commit()
        print("✅ 업데이트 완료.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    update_manutd_manager()
