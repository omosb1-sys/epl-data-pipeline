import os
import mysql.connector

def correct_transfers():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password=os.getenv("MYSQL_PASSWORD", ""),
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )
    cursor = conn.cursor()
    
    # [정정] 아스날이 요케레스 영입 성공!
    # 아스날: 이삭 -> 요케레스 교체 (이삭은 너무 비싸서 포기했다고 가정)
    # 맨유: 요케레스 -> 세슈코 교체 (대안 영입)
    
    updates = {
        "아스날": "빅토르 요케레스, 마르틴 수비멘디, 르로이 사네",
        "맨체스터 유나이티드": "벤자민 세슈코, 알폰소 데이비스, 프랑코 마스탄투오노"
    }
    
    print("🚑 이적시장 오피셜 정정 중...")
    
    for team, new_in in updates.items():
        sql = "UPDATE clubs SET transfers_in = %s WHERE team_name LIKE %s"
        cursor.execute(sql, (new_in, f"%{team}%"))
        print(f"✅ {team}: 영입 명단 수정 완료 -> {new_in}")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    correct_transfers()
