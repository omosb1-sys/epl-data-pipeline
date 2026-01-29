import mysql.connector

def update_realtime_stats():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="0623os1978",
            database="epl_x_db",
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()

        # [1] 부상 현황 업데이트 (2026년 1월 실시간 정보 반영)
        print("🚑 부상 현황 업데이트 중...")
        
        # 토트넘: 주전 줄부상 (매디슨, 판더벤, 로메로, 솔란케 등)
        cursor.execute("UPDATE clubs SET injury_level = '주전 줄부상 비상' WHERE team_name = '토트넘 홋스퍼'")
        
        # 맨시티: 로드리 시즌 아웃, 스톤스, 코바치치
        cursor.execute("UPDATE clubs SET injury_level = '주전 줄부상 비상' WHERE team_name = '맨체스터 시티'")
        
        # 아스날: 칼라피오리, 모스케라
        cursor.execute("UPDATE clubs SET injury_level = '심각' WHERE team_name = '아스날'")
        
        # 첼시: 리바이 콜윌 시즌 아웃, 라비아
        cursor.execute("UPDATE clubs SET injury_level = '심각' WHERE team_name = '첼시'")
        
        # 맨유: 요로, 린델뢰프
        cursor.execute("UPDATE clubs SET injury_level = '경미' WHERE team_name = '맨체스터 유나이티드'")

        # [2] 겨울 이적시장 루머 업데이트
        print("❄️ 겨울 이적시장 루머 업데이트 중...")
        
        # 아스날: 요케레스(Gyokeres) 영입설
        cursor.execute("UPDATE clubs SET winter_rumors_in = '빅토르 요케레스(85%)' WHERE team_name = '아스날'")
        
        # 맨시티: 수비형 미들 보강설
        cursor.execute("UPDATE clubs SET winter_rumors_in = '마르틴 수비멘디(60%), 에데르송(아탈란타)(50%)' WHERE team_name = '맨체스터 시티'")

        conn.commit()
        print("✅ 실시간 데이터가 DB에 반영되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    update_realtime_stats()
