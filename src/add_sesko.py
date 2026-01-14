import mysql.connector

def add_sesko_official():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )
    cursor = conn.cursor()
    
    # 2025년 맨유 오피셜 영입: 세슈코 포함
    # 기존 오피셜 멤버 유지 + 세슈코 추가
    # (사용자 피드백 반영: 세슈코는 2025년 영입 확정)
    
    manutd_in = "벤자민 세슈코, 레니 요로, 조슈아 지르크지, 마타이스 데 리흐트, 누사이르 마즈라위, 마누엘 우가르테"
    
    sql = "UPDATE clubs SET transfers_in = %s WHERE team_name LIKE '맨체스터 유나이티드'"
    cursor.execute(sql, (manutd_in,))
    
    print(f"✅ 맨유 영입 명단 업데이트 완료 (세슈코 포함): {manutd_in}")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    add_sesko_official()
