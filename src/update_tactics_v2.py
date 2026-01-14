import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def update_tactics_future_ver():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 2025년 12월 기준 (사용자 시나리오 반영)
    # 토트넘: 포스테코글루 경질 -> 후임 경질 -> 현재 혼란기
    
    updates = {
        "맨체스터 유나이티드": {
            "fmt": "3-4-3 (아모림 2년차)",
            "desc": "후뱅 아모림 감독의 시스템이 완벽히 정착되었습니다. 요케레스와 세슈코 투톱을 활용한 파괴적인 공격력이 특징.",
            "w_in": "디오고 코스타(GK, 60%), 누누 멘데스(40%)",
            "w_out": "안토니(99%), 카세미루(80%)"
        },
        "토트넘 홋스퍼": {
            "fmt": "4-2-3-1 (대행 체제)",
            "desc": "포스테코글루와 후임 감독이 연이어 경질된 후, **라이언 메이슨 대행**이 수습 중입니다. 수비 안정화를 최우선으로 하며 역습을 노립니다.",
            "w_in": "[감독] 토마스 프랭크(유력), [감독] 에디 하우(링크)",
            "w_out": "손흥민(계약 만료 임박설), 매디슨(불화설)"
        },
        "아스날": {
            "fmt": "4-3-3 (아르테타)",
            "desc": "완성형에 도달한 아르테타볼. 리그 1위를 질주하며 챔스 우승까지 노리는 완벽한 밸런스를 보여줍니다.",
            "w_in": "비르츠(레알과 경쟁 50%)",
            "w_out": "조르지뉴(은퇴 예정)"
        },
        "맨체스터 시티": {
            "fmt": "4-1-4-1 (포스트 펩?)",
            "desc": "과르디올라의 거취가 불분명한 가운데, 홀란드의 득점력에 의존하는 경향이 강해졌습니다.",
            "w_in": "루카스 파케타(재점화 70%)",
            "w_out": "데 브라위너(사우디 90%)"
        },
        "첼시": {
            "fmt": "4-2-3-1 (마레스카)",
            "desc": "마레스카 감독 하에서 젊은 선수들의 조직력이 살아났습니다. 팔머가 에이스 놀이를 하고 있습니다.",
            "w_in": "빅터 오시멘(겨울 영입 유력)",
            "w_out": "무드릭(임대)"
        },
        "리버풀": {
            "fmt": "4-3-3 (슬롯)",
            "desc": "아르네 슬롯 감독의 2번째 시즌. 살라의 대체자를 찾는 것이 최대 과제입니다.",
            "w_in": "쿠보 타케후사(40%), 아데예미(60%)",
            "w_out": "살라(사우디 이적설)"
        }
    }
    
    # 나머지 구단은 기본값 유지하되, 2025년 느낌으로
    default_fmt = "4-4-2"
    default_desc = "치열한 중위권 싸움 중입니다."
    
    print("👔 [2025-26 시즌] 감독 잔혹사 및 전술 데이터 반영 중...")
    
    cursor.execute("SELECT team_name FROM clubs")
    all_teams = [row[0] for row in cursor.fetchall()]
    
    for team in all_teams:
        data = updates.get(team)
        if data:
            fmt, desc, win, wout = data['fmt'], data['desc'], data['w_in'], data['w_out']
        else:
            fmt, desc, win, wout = default_fmt, default_desc, "정보 없음", "정보 없음"
            
        sql = """
            UPDATE clubs 
            SET tactics_formation = %s, tactics_desc = %s, winter_rumors_in = %s, winter_rumors_out = %s
            WHERE team_name = %s
        """
        cursor.execute(sql, (fmt, desc, win, wout, team))
        
    conn.commit()
    conn.close()
    print("✨ 2025년 12월 기준 데이터 업데이트 완료!")

if __name__ == "__main__":
    update_tactics_future_ver()
