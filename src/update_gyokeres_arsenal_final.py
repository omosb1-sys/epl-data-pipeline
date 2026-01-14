import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def finalize_2025_transfers():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 1. 아스날: 빅토르 요케레스 영입 확정 (24/25~25/25 리얼타임 반영)
    arsenal_in = "빅토르 요케레스, 리카르도 칼라피오리, 미켈 메리노, 라힘 스털링(임대), 다비드 라야"
    arsenal_tactics = "미켈 아르테타의 4-3-3 완성형. 최전방의 빅토르 요케레스가 압도적인 피지컬과 결정력을 제공하며 박스 안에서의 파괴력을 극대화함."
    
    cursor.execute("""
        UPDATE clubs 
        SET transfers_in = %s, tactics_desc = %s, power_index = 96
        WHERE team_name = '아스날'
    """, (arsenal_in, arsenal_tactics))

    # 2. 맨유: 벤자민 세슈코 영입 확정 (영입설에서 오피셜로 이동)
    manutd_in = "벤자민 세슈코, 레니 요로, 조슈아 지르크지, 마타이스 데 리흐트, 누사이르 마즈라위, 마누엘 우가르테"
    # 겨울 루머에서는 요케레스 삭제, 새로운 타겟 추가
    manutd_rumors = "디오고 코스타(GK, 60%), 누누 멘데스(40%), 에반 퍼거슨(20%)"
    manutd_tactics = "후뱅 아모림의 3-4-3 시스템. 벤자민 세슈코를 타겟맨으로 활용하며 윙백들의 높은 전진성을 바탕으로 한 공격적인 축구."

    cursor.execute("""
        UPDATE clubs 
        SET transfers_in = %s, winter_rumors_in = %s, tactics_desc = %s, power_index = 89
        WHERE team_name = '맨체스터 유나이티드'
    """, (manutd_in, manutd_rumors, manutd_tactics))

    print("✅ 2025 타임라인 강제 동기화 완료!")
    print("✅ 아스날: 빅토르 요케레스 (오피셜)")
    print("✅ 맨유: 벤자민 세슈코 (오피셜), 요케레스 루머 삭제")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    finalize_2025_transfers()
