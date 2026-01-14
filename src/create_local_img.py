from PIL import Image, ImageDraw, ImageFont
import os

def create_manutd_image():
    # 1. 붉은색 배경 이미지 생성 (맨유 컬러 #DA291C)
    width, height = 800, 600
    color = (218, 41, 28) # Man Utd Red
    img = Image.new('RGB', (width, height), color)
    
    # 2. 텍스트 추가 (폰트가 없으면 기본으로)
    draw = ImageDraw.Draw(img)
    
    # 중앙에 텍스트 쓰기에 적당한 위치 계산 (대충 정중앙)
    text = "THEATRE OF DREAMS\nOld Trafford"
    
    # 텍스트 그리기 (기본 폰트라 좀 작을 수 있음, 그래도 없는것보단 낫다)
    # PIL 기본 폰트는 사이즈 조절이 안되므로, 도형으로 장식
    draw.rectangle([200, 250, 600, 350], outline="white", width=5)
    draw.text((350, 280), "Old Trafford", fill="white")
    draw.text((330, 310), "Manchester United", fill="white")
    
    # 3. 파일로 저장
    if not os.path.exists("stadiums"):
        os.makedirs("stadiums")
        
    save_path = "stadiums/man_utd.jpg"
    img.save(save_path)
    print(f"✅ [성공] 맨유 전용 로컬 이미지 생성 완료: {save_path}")

if __name__ == "__main__":
    create_manutd_image()
