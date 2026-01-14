import os
import requests
import mysql.connector

# 이미지 다운로드 및 DB 업데이트 스크립트

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0623os1978",
        database="epl_x_db",
        auth_plugin='mysql_native_password'
    )

def download_and_update():
    # 1. 다운로드할 이미지 목록 (안정적인 소스 사용)
    images = {
        "맨체스터 유나이티드": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/3/33/Manchester_United_Old_Trafford_Stadium.jpg",
            "filename": "man_utd.jpg"
        },
        "맨체스터 시티": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/b/b8/Etihad_Stadium.jpg",
            "filename": "man_city.jpg"
        },
        "아스날": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/1/14/Emirates_Stadium_east_side_at_dusk.jpg",
            "filename": "arsenal.jpg"
        },
        "리버풀": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/0/02/Panorama_of_Anfield_with_new_main_stand_%2829676137824%29.jpg",
            "filename": "liverpool.jpg"
        },
        "첼시": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/3/36/Stamford_Bridge_%28Chelsea_FC%29.jpg",
            "filename": "chelsea.jpg"
        },
        "토트넘 홋스퍼": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/4/43/Tottenham_Hotspur_Stadium_July_2021.jpg",
            "filename": "tottenham.jpg"
        }
    }
    
    conn = connect_to_db()
    cursor = conn.cursor()
    
    print("📥 구장 이미지 로컬 다운로드 시작...")
    
    for team, info in images.items():
        try:
            # 이미지 다운로드 (User-Agent 헤더 추가로 차단 방지)
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(info['url'], headers=headers, timeout=10)
            
            if response.status_code == 200:
                # 파일로 저장
                filepath = f"stadiums/{info['filename']}"
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # DB 업데이트 (로컬 경로로)
                # Streamlit에서 로컬 이미지는 그냥 파일명만 있으면 되는 경우가 많으나,
                # app.py 로직에 맞춰서 처리 필요. 여기서는 일단 경로 저장.
                sql = "UPDATE clubs SET stadium_img = %s WHERE team_name LIKE %s"
                cursor.execute(sql, (filepath, f"%{team}%")) # %검색% 사용
                print(f"✅ {team}: 다운로드 및 DB 업데이트 완료 ({filepath})")
            else:
                print(f"⚠️ {team}: 다운로드 실패 (Status {response.status_code})")
                
        except Exception as e:
            print(f"❌ {team}: 오류 발생 - {e}")
            
    conn.commit()
    conn.close()
    print("✨ 모든 작업 완료!")

if __name__ == "__main__":
    download_and_update()
