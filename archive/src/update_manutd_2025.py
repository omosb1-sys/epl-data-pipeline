import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def update_manutd_2025():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 막내님이 주신 2025년 여름 이적시장 디테일
    t_in = """[영입]
- 마테우스 쿠냐 (FW, 울버햄튼, £62.5m)
- 벤자민 세슈코 (FW, 라이프치히, £66.3m)
- 브라이언 음뵈모 (FW, 브렌트포드, £65m)
- 센네 라멘스 (GK, 앤트워프, £18.1m)
- 디에고 레온 (DF, 세로 포르테뇨, £6m)
- 엔조 카나-비익 (FW, 르 아브르, FA)
- 할리 엠스덴-제임스 (DF, 사우스햄튼, FA)"""

    t_out = """[방출/임대]
- 마커스 래시포드 (→ 바르셀로나, 임대)
- 알레한드로 가르나초 (→ 첼시, 완전 이적)
- 안토니 (→ 레알 베티스, 완전 이적)
- 라스무스 호일룬 (→ 나폴리, 임대)
- 제이든 산초 (→ 아스톤 빌라, 임대)
- 안드레 오나나 (→ 트라브존스포르, 임대)
- 빅토르 린델로프 (→ 아스톤 빌라, FA)
- 크리스티안 에릭센 (계약 만료)
- 조니 에반스 (계약 만료)"""

    # 업데이트 쿼리
    sql = "UPDATE clubs SET transfers_in=%s, transfers_out=%s WHERE team_name='맨체스터 유나이티드'"
    
    try:
        cursor.execute(sql, (t_in, t_out))
        conn.commit()
        print("✅ 맨체스터 유나이티드 2025 이적시장 데이터 업데이트 완료!")
        print("   - 쿠냐, 세슈코 IN / 래시포드, 가르나초 OUT 적용됨.")
    except Exception as e:
        print(f"❌ 업데이트 실패: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    update_manutd_2025()
