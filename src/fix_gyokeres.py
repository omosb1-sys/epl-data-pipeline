import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def fix_gyokeres_saga():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 1. 아스날: 요케레스 영입 확정 (오피셜 명단 추가)
    # 기존 아스날 영입 명단(transfers_in) 앞에 '빅토르 요케레스' 추가
    # 요케레스가 아스날의 핵심 스트라이커임을 명시
    
    # 아스날 전술 설명도 업데이트 (요케레스 활용)
    sql_ars_tac = """
        UPDATE clubs 
        SET tactics_desc = '미켈 아르테타의 4-3-3 완성형. **최전방의 요케레스**가 버텨주고 2선 침투로 득점을 노리는 파괴적인 시스템.',
            transfers_in = CONCAT('빅토르 요케레스(Target Man), ', transfers_in)
        WHERE team_name = '아스날'
    """
    cursor.execute(sql_ars_tac)
    
    # 2. 맨유: 요케레스에 대한 미련(겨울 루머) 삭제
    # 요케레스는 이미 아스날 갔으니 맨유 겨울 루머에서 제거해야 함.
    # 대신 다른 현실적인 타겟(예: 에반 퍼거슨 등)으로 대체하거나 비워둠.
    
    sql_man_rumor = """
        UPDATE clubs 
        SET winter_rumors_in = '디오고 코스타(GK, 60%), 누누 멘데스(40%), 에반 퍼거슨(20%)'
        WHERE team_name = '맨체스터 유나이티드'
    """
    cursor.execute(sql_man_rumor)

    print("✅ 아스날: 빅토르 요케레스 영입 확정 및 전술 반영 완료.")
    print("✅ 맨유: 요케레스 영입설 삭제 (현실화).")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    fix_gyokeres_saga()
