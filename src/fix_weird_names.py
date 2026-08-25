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

def fix_weird_team_names():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # [오류 이름] -> [정상 이름] 매핑
    corrections = {
        "운동하는 햄": "웨스트햄 유나이티드",
        "토트넘넘 공유스퍼": "토트넘 홋스퍼",
        "위치 위치": "입스위치 타운",
        "크리스탈 블록": "크리스탈 팰리스",
        "지금": "노팅엄 포레스트",  # Nottingham -> Now(지금)?? 번역 오류 추정
        "가": "아스날",      # 가..너? Gunners?
        "빌라": "아스톤 빌라",
        "늑대": "울버햄튼 원더러스",
        "갈매기": "브라이튼 앤 호브 알비온",
        "벌": "브렌트포드",  # Bees -> 벌
        "사탕": "에버튼",    # Toffees -> 사탕
        "성도": "사우스햄튼", # Saints -> 성도
        "여우": "레스터 시티" # Foxes -> 여우
    }
    
    # 혹시 모르니 영문 팀명이나 기존 키워드로도 정상화 시도
    # 여기서는 '운동하는 햄' 처럼 완전히 식별 가능한 오류 이름은 직접 수정
    
    print("🚑 팀 이름 긴급 복구 작전 시작...")
    
    updated_count = 0
    
    # 1. 명확한 오류 이름 수정 (직접 매핑)
    for weird, correct in corrections.items():
        # 해당 이름이 있는지 확인 후 업데이트
        check_sql = "SELECT id FROM clubs WHERE team_name = %s"
        cursor.execute(check_sql, (weird,))
        if cursor.fetchone():
            sql = "UPDATE clubs SET team_name = %s WHERE team_name = %s"
            cursor.execute(sql, (correct, weird))
            print(f"✅ 수정 완료: {weird} -> {correct}")
            updated_count += 1
            
    # 2. 혹시 모르니 정상 이름 리스트로 강제 초기화 (안전장치)
    # 기존 DB의 ID 순서나 특정 키워드를 기반으로 복구
    # (일단 위 1번 단계로 해결되는지 보고, 안되면 2번 실행)

    if updated_count == 0:
        print("⚠️ 오염된 팀 이름을 찾지 못했습니다. DB 상태를 직접 확인해야 합니다.")
    else:
        print(f"✨ 총 {updated_count}개 팀 이름 정상화 완료!")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    fix_weird_team_names()
