import mysql.connector
import os

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=os.getenv("MYSQL_PASSWORD", ""),
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def map_images_to_db():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 파일명 -> 팀명 매핑 (파일명에 맞춰서 팀명 지정)
    # LIKE 검색을 위해 팀명의 일부만 사용
    mapping = {
        "arsenal.png": "아스날",
        "bournemouth.png": "본머스",
        "brentford.png": "브렌트포드",
        "brighton_h_a.png": "브라이튼",
        "chelsea.png": "첼시",
        "crystal_p.png": "크리스탈",
        "everton.png": "에버턴",
        "fulham.png": "풀럼",
        "leichester_c.png": "레스터",
        "liverpool.jpg": "리버풀",
        "man_city.jpg": "맨체스터 시티",
        "man_utd.jpg": "맨체스터 유나이티드",
        "newcastle_u.png": "뉴캐슬",
        "nottingham_f.png": "노팅엄",
        "s_hampton.png": "사우샘프턴",
        "totten_h.png": "토트넘",
        "west.h.png": "웨스트햄",
        "wolverhampton_w.png": "울버햄튼"
    }
    
    print("🔄 구단 이미지 DB 연결 작업 시작...")
    
    updated_count = 0
    
    for filename, team_keyword in mapping.items():
        # 파일이 실제로 존재하는지 확인
        if os.path.exists(f"stadiums/{filename}"):
            path = f"stadiums/{filename}"
            
            # DB 업데이트
            sql = "UPDATE clubs SET stadium_img = %s WHERE team_name LIKE %s"
            cursor.execute(sql, (path, f"%{team_keyword}%"))
            
            if cursor.rowcount > 0:
                print(f"✅ {team_keyword} -> {path} 연결 완료")
                updated_count += 1
            else:
                print(f"⚠️ {team_keyword} 팀을 DB에서 찾을 수 없음")
        else:
            print(f"❌ {filename} 파일이 없습니다. (스킵)")
            
    conn.commit()
    conn.close()
    print(f"\n✨ 총 {updated_count}개 구단 이미지 연결 완료!")

if __name__ == "__main__":
    map_images_to_db()
