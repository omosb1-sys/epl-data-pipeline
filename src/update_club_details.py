import os
import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=os.getenv("MYSQL_PASSWORD", ""),
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def update_schema_and_data():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    print("🛠️ DB 리모델링 시작 (테이블 확장 중...)")
    
    # 1. 테이블 컬럼 추가 (ALTER TABLE)
    # 이미 존재할 수 있으므로 안전하게 하나씩 시도
    columns = [
        "ADD COLUMN stadium_img VARCHAR(255)",
        "ADD COLUMN club_value VARCHAR(50)",
        "ADD COLUMN current_rank INT DEFAULT 0",
        "ADD COLUMN last_season_rank INT DEFAULT 0",
        "ADD COLUMN matches_played INT DEFAULT 0",
        "ADD COLUMN wins INT DEFAULT 0",
        "ADD COLUMN draws INT DEFAULT 0",
        "ADD COLUMN losses INT DEFAULT 0",
        "ADD COLUMN transfers_in TEXT",  # 줄바꿈이 있는 긴 텍스트
        "ADD COLUMN transfers_out TEXT"
    ]
    
    for col_sql in columns:
        try:
            cursor.execute(f"ALTER TABLE clubs {col_sql}")
        except mysql.connector.Error as err:
            # 컬럼이 이미 있으면 1060 에러가 뜹니다. 무시하고 진행.
            if err.errno != 1060:
                print(f"⚠️ 컬럼 추가 중 알림: {err}")

    print("✅ 테이블 확장 완료! 데이터 업데이트 시작...")

    # 2. 팀별 상세 데이터 업데이트 (주요 6개 팀 예시)
    updates = [
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Old_Trafford%2C_Manchester.jpg/600px-Old_Trafford%2C_Manchester.jpg",
            "$6.55B (포브스 2위)", 
            8, 8, # 현재 8위, 지난 시즌 8위
            17, 7, 5, 5, # 17경기 7승 5무 5패
            "레니 요로 (62m€)\n마누엘 우가르테 (50m€)\n조슈아 지르크지 (42m€)\n마타이스 더 리흐트 (45m€)",
            "스콧 맥토미니 (나폴리)\n아론 완비사카 (웨스트햄)\n메이슨 그린우드 (마르세유)",
            "맨체스터 유나이티드"
        ),
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Etihad_Stadium.jpg/600px-Etihad_Stadium.jpg",
            "$5.1B (포브스 5위)", 
            2, 1, # 현재 2위, 지난 시즌 우승
            17, 12, 3, 2,
            "사비뉴 (25m€)\n일카이 귄도안 (FA)",
            "훌리안 알바레즈 (AT마드리드)\n주앙 칸셀루 (알 힐랄)",
            "맨체스터 시티"
        ),
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Emirates_Stadium_east_side_at_dusk.jpg/600px-Emirates_Stadium_east_side_at_dusk.jpg",
            "$2.6B (포브스 8위)", 
            3, 2,
            17, 10, 5, 2,
            "리카르도 칼라피오리 (45m€)\n미켈 메리노 (32m€)",
            "에밀 스미스 로우 (풀럼)\n에디 은케티아 (팰리스)",
            "아스날"
        ),
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Panorama_of_Anfield_with_new_main_stand_%2829676137824%29.jpg/600px-Panorama_of_Anfield_with_new_main_stand_%2829676137824%29.jpg",
            "$5.37B (포브스 4위)", 
            1, 3,
            17, 13, 3, 1,
            "페데리코 키에사 (12m€)\n기오르기 마마르다슈빌리 (임대)",
            "티아고 알칸타라 (은퇴)\n조엘 마팁 (FA)",
            "리버풀"
        ),
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Stamford_Bridge_%28Chelsea_FC%29.jpg/600px-Stamford_Bridge_%28Chelsea_FC%29.jpg",
            "$3.1B (포브스 7위)", 
            4, 6,
            17, 9, 4, 4,
            "페드로 네투 (60m€)\n주앙 펠릭스 (52m€)\n제이든 산초 (임대)",
            "코너 갤러거 (AT마드리드)\n라힘 스털링 (임대)",
            "첼시"
        ),
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Tottenham_Hotspur_Stadium_July_2021.jpg/600px-Tottenham_Hotspur_Stadium_July_2021.jpg",
            "$3.2B (포브스 6위)", 
            7, 5,
            17, 8, 2, 7,
            "도미닉 솔란케 (64m€)\n아치 그레이 (41m€)",
            "해리 케인 (작년 이적)\n에릭 다이어 (뮌헨)",
            "토트넘 홋스퍼"
        )
    ]
    
    sql_update = """
        UPDATE clubs 
        SET stadium_img=%s, club_value=%s, current_rank=%s, last_season_rank=%s,
            matches_played=%s, wins=%s, draws=%s, losses=%s,
            transfers_in=%s, transfers_out=%s
        WHERE team_name=%s
    """
    
    cursor.executemany(sql_update, updates)
    conn.commit()
    conn.close()
    
    print(f"✨ {cursor.rowcount}개 구단의 상세 정보(구단가치, 이적시장 등) 업데이트 완료!")

if __name__ == "__main__":
    update_schema_and_data()
