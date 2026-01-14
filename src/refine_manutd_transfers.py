import mysql.connector

def fix_manutd_transfers():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )
    cursor = conn.cursor()
    
    # 맨유 영입 명단 현실성 보정
    # 알폰소 데이비스(레알 유력) -> 브렌스웨이트 (맨유 강력 링크)
    # 마스탄투오노 -> 케르케즈 (레프트백 보강)
    
    new_in = "벤자민 세슈코, 재러드 브렌스웨이트, 밀로스 케르케즈"
    
    sql = "UPDATE clubs SET transfers_in = %s WHERE team_name LIKE '맨체스터 유나이티드'"
    cursor.execute(sql, (new_in,))
    
    print(f"✅ 맨유 영입 명단 수정 완료: {new_in}")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    fix_manutd_transfers()
