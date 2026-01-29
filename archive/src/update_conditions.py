import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def update_team_conditions():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 1. 테이블에 컨디션 관련 컬럼 추가 (없으면 생성)
    try:
        cursor.execute("ALTER TABLE clubs ADD COLUMN injury_level VARCHAR(50) DEFAULT '보통'")
        cursor.execute("ALTER TABLE clubs ADD COLUMN rest_days INT DEFAULT 3")
        cursor.execute("ALTER TABLE clubs ADD COLUMN team_mood VARCHAR(50) DEFAULT '보통'")
        print("✅ DB 구조 업그레이드 완료 (부상/휴식/분위기 컬럼 추가)")
    except:
        print("ℹ️ 컬럼이 이미 존재하거나 오류 발생 (넘어감)")
        
    # 2. 각 팀별 현재 컨디션 데이터 입력 (2025년 12월 가상 시나리오)
    # [부상수준, 휴식일, 분위기]
    # 부상수준: '풀전력', '경미', '심각'
    # 분위기: '최상', '좋음', '보통', '나쁨', '최악'
    
    conditions = {
        "맨체스터 유나이티드": ["심각", 3, "좋음"], # 부상은 많으나 최근 연승으로 분위기 좋음
        "맨체스터 시티": ["경미", 4, "나쁨"],      # 로드리 부상 여파, 분위기 다운
        "아스날": ["풀전력", 7, "최상"],          # 부상 없음, 1주일 휴식, 1위 질주
        "리버풀": ["보통", 3, "좋음"],            # 살라 공백 있으나 잇몸으로 버팀
        "토트넘 홋스퍼": ["심각", 2, "최악"],     # 줄부상 + 2일만 휴식 + 연패
        "첼시": ["풀전력", 5, "보통"],
        "뉴캐슬 유나이티드": ["보통", 4, "좋음"],
        "아스톤 빌라": ["경미", 3, "나쁨"],
        "웨스트햄 유나이티드": ["풀전력", 6, "보통"],
        "울버햄튼": ["심각", 3, "나쁨"],          # 황희찬 부상 등
        "브라이튼": ["보통", 4, "좋음"],
        "에버튼": ["경미", 5, "최악"],            # 강등권 싸움
        "레스터 시티": ["풀전력", 7, "보통"],
        "크리스탈 팰리스": ["보통", 4, "좋음"],
        "브렌트포드": ["심각", 3, "나쁨"],
        "노팅엄 포레스트": ["경미", 5, "보통"],
        "풀럼": ["풀전력", 6, "좋음"],
        "본머스": ["보통", 4, "보통"],
        "사우스햄튼": ["심각", 2, "최악"],        # 꼴찌 + 부상
        "입스위치 타운": ["경미", 3, "나쁨"]
    }
    
    print("🚑 전 구단 부상/컨디션 데이터 업데이트 중...")
    
    for team, (inj, rest, mood) in conditions.items():
        sql = """
            UPDATE clubs 
            SET injury_level = %s, rest_days = %s, team_mood = %s
            WHERE team_name LIKE %s
        """
        cursor.execute(sql, (inj, rest, mood, f"%{team}%"))
        
    conn.commit()
    conn.close()
    print("✨ 모든 팀의 실시간 컨디션 정보 반영 완료!")

if __name__ == "__main__":
    update_team_conditions()
