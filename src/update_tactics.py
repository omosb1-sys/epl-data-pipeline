import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def update_tactics_and_rumors():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 1. 컬럼 추가 (전술/겨울이적 관련)
    try:
        cursor.execute("ALTER TABLE clubs ADD COLUMN tactics_formation VARCHAR(20) DEFAULT '4-4-2'")
        cursor.execute("ALTER TABLE clubs ADD COLUMN tactics_desc TEXT")
        cursor.execute("ALTER TABLE clubs ADD COLUMN winter_rumors_in TEXT") # 영입 소문
        cursor.execute("ALTER TABLE clubs ADD COLUMN winter_rumors_out TEXT") # 방출 소문
        print("✅ DB 구조 업그레이드 완료 (전술 및 겨울이적예측 컬럼)")
    except:
        print("ℹ️ 컬럼이 이미 존재합니다.")

    # 2. 데이터 매핑 (2025 기준 가상/현실 반영)
    updates = {
        "맨체스터 유나이티드": {
            "fmt": "3-4-3 (아모림)",
            "desc": "후뱅 아모림 감독 특유의 **3백 시스템**. 윙백의 높은 공격 가담과 전방의 강한 압박을 중시합니다. 빠른 공수 전환이 핵심.",
            "w_in": "빅토르 요케레스(90%), 알폰소 데이비스(60%)",
            "w_out": "해리 매과이어(80%), 크리스티안 에릭센(70%)"
        },
        "맨체스터 시티": {
            "fmt": "3-2-4-1",
            "desc": "펩 과르디올라의 **인버티드 풀백** 전술. 존 스톤스를 미드필더로 올리며 중원 수적 우위를 점유하고, 홀란드를 고립시키지 않는 유기적인 패스 워크.",
            "w_in": "자말 무시알라(40%), 플로리안 비르츠(35%)",
            "w_out": "카일 워커(50%)"
        },
        "아스날": {
            "fmt": "4-3-3",
            "desc": "미켈 아르테타의 **유기적인 포지셔닝**. 외데고르를 중심으로 한 하프스페이스 공략과 세트피스에서의 강력한 득점력이 특징.",
            "w_in": "이삭(뉴캐슬)(30%), 르로이 사네(20%)",
            "w_out": "토마스 파티(60%), 제수스(40%)"
        },
        "리버풀": {
            "fmt": "4-2-3-1 (슬롯)",
            "desc": "아르네 슬롯 감독의 **통제된 카오스**. 클롭 시절보다 조금 더 점유율을 중시하지만, 여전히 측면에서의 폭발적인 스피드를 활용합니다.",
            "w_in": "추아메니(20%), 로익 바데(40%)",
            "w_out": "엔도 와타루(80%)"
        },
        "토트넘 홋스퍼": {
            "fmt": "4-3-3 (엔지볼)",
            "desc": "포스테코글루의 **극단적인 공격 축구**. 라인을 하프라인까지 올리고 풀백이 중앙으로 들어오는 인버티드 움직임을 가져갑니다. '뒤는 없다'는 식의 닥공.",
            "w_in": "앙헬 고메스(50%), 에베레치 에제(40%)",
            "w_out": "히샬리송(60%), 레길론(90%)"
        },
        "첼시": {
            "fmt": "4-2-3-1 (마레스카)",
            "desc": "엔조 마레스카의 **점유율 중심**. 펩의 제자답게 후방 빌드업을 극도로 중요시하며, 팔머에게 프리롤을 부여해 창의성을 극대화합니다.",
            "w_in": "오시멘(겨울 재도전 50%)",
            "w_out": "칠웰(90%), 무드릭(임대설)"
        },
        "뉴캐슬 유나이티드": {
            "fmt": "4-3-3",
            "desc": "에디 하우의 **강력한 피지컬 압박**. 중원에서의 거친 싸움과 앤서니 고든의 빠른 역습을 주 무기로 삼습니다.",
            "w_in": "마크 게히(재도전), 음뵈모",
            "w_out": "알미론, 트리피어"
        },
        "아스톤 빌라": {
            "fmt": "4-4-2 / 4-2-3-1",
            "desc": "우나이 에메리의 **치밀한 오프사이드 트랩**. 수비 라인을 극도로 정교하게 컨트롤하며, 왓킨스를 활용한 직선적인 역습이 매섭습니다.",
            "w_in": "페란 토레스, 루크만",
            "w_out": "디에고 카를로스"
        }
    }
    
    # 나머지 팀들은 기본값으로 채움
    default_fmt = "4-4-2 (Balance)"
    default_desc = "안정적인 수비 블록을 형성한 후, 측면을 활용한 빠른 역습을 노리는 전형적인 카운터 어택 전술입니다."
    
    print("👔 감독 전술 및 겨울 이적 루머 데이터 업데이트 중...")
    
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
    print("✨ 모든 전술/루머 데이터 반영 완료!")

if __name__ == "__main__":
    update_tactics_and_rumors()
