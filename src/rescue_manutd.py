import os
import requests

def rescue_manutd():
    # Unsplash의 튼튼한 붉은 경기장 이미지
    url = "https://images.unsplash.com/photo-1522778119026-d647f0565c6a?auto=format&fit=crop&w=800&q=80"
    file_path = "stadiums/man_utd.jpg"
    
    print(f"🚑 맨유 이미지 긴급 구조 시작... ({url})")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ 구조 성공! 로컬 파일 생성됨: {file_path}")
        else:
            print(f"❌ 구조 실패: 상태 코드 {response.status_code}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    rescue_manutd()
