import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os
import glob
from sklearn.cluster import KMeans
import colorsys
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 준비
data_dir = "./naver_shopping_analysis/data"
csv_files = glob.glob(os.path.join(data_dir, "naver_shopping_*.csv"))
df = pd.concat([pd.read_csv(f, encoding='utf-8-sig') for f in csv_files], ignore_index=True)

# 전처리
df = df.drop_duplicates(subset=['product_id']).copy()
df['lprice'] = pd.to_numeric(df['lprice'], errors='coerce')
df = df.dropna(subset=['lprice', 'image'])

# 샘플링: 가격대별 골고루 60개 추출 (High, Mid, Low)
df = df.sort_values('lprice', ascending=False)
high = df.head(20)
low = df.tail(20)
mid = df.iloc[len(df)//2-10 : len(df)//2+10]
sample_df = pd.concat([high, mid, low])

print(f"📸 이미지 분석 대상: {len(sample_df)}개 샘플 다운로드 및 분석 시작...")

def get_image_features(url):
    try:
        response = requests.get(url, timeout=3)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((100, 100)) # 속도를 위해 리사이징
        
        # 1. 밝기 (Brightness)
        # Grayscale 변환 후 평균 픽셀 값
        gray_img = img.convert('L')
        brightness = np.mean(np.array(gray_img))
        
        # 2. 주요 색상 (Dominant Color) & 채도 (Saturation)
        # K-Means로 주요 색상 1개 추출 (가장 많이 쓰인 색)
        ar = np.array(img).reshape(-1, 3)
        kmeans = KMeans(n_clusters=1, n_init=5).fit(ar)
        dominant_rgb = kmeans.cluster_centers_[0]
        
        # RGB -> HSV 변환 (채도 분석용)
        # colorsys는 0~1 입력을 받음
        h, s, v = colorsys.rgb_to_hsv(dominant_rgb[0]/255, dominant_rgb[1]/255, dominant_rgb[2]/255)
        
        return pd.Series([brightness, s, v])
    except Exception as e:
        return pd.Series([np.nan, np.nan, np.nan])

# 이미지 분석 실행
features = sample_df['image'].apply(get_image_features)
features.columns = ['brightness', 'saturation', 'value']

sample_df = pd.concat([sample_df, features], axis=1)
sample_df = sample_df.dropna(subset=['brightness'])

print("✅ 이미지 분석 완료. 시각화 생성 중...")

# 2. 분석 결과 시각화
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 시각화 1: 채도(Saturation)와 가격의 관계 (파스텔톤(저채도) vs 비비드(고채도))
# 채도가 낮을수록(0.0~0.4) 파스텔/흰색/회색 계열, 높을수록(0.7~1.0) 원색
plt.figure(figsize=(10, 6))
sns.scatterplot(x='saturation', y='lprice', data=sample_df, hue='lprice', palette='coolwarm', s=100)
plt.axvline(x=0.4, color='gray', linestyle='--')
plt.text(0.1, sample_df['lprice'].max(), 'Pastel/Modern Zone', fontsize=12, color='blue')
plt.text(0.6, sample_df['lprice'].max(), 'Vivid/Complex Zone', fontsize=12, color='red')
plt.title('이미지 톤(채도)에 따른 가격 포지셔닝')
plt.xlabel('이미지 채도 (Saturation, 0:무채색 -> 1:원색)')
plt.ylabel('가격 (Price)')
plt.savefig('image_saturation_price.png')

# 시각화 2: 밝기(Brightness)와 가격
plt.figure(figsize=(10, 6))
sns.regplot(x='brightness', y='lprice', data=sample_df, scatter_kws={'s':50, 'alpha':0.6}, line_kws={'color':'red'})
plt.title('이미지 밝기(Brightness)와 가격의 상관관게')
plt.xlabel('이미지 밝기 (0:어두움 -> 255:밝음)')
plt.savefig('image_brightness_price.png')

# 상관계수 출력
corr_sat = sample_df['saturation'].corr(sample_df['lprice'])
corr_bri = sample_df['brightness'].corr(sample_df['lprice'])

print(f"\n🎨 [Visual Pricing Insight]")
print(f"1. 채도(Saturation)와 가격 상관계수: {corr_sat:.3f}")
print(f"   -> {'음의 상관관계(채도가 낮을수록 비쌈)' if corr_sat < 0 else '양의 상관관계'}")
print(f"2. 밝기(Brightness)와 가격 상관계수: {corr_bri:.3f}")
print(f"   -> {'밝을수록 비싼 경향' if corr_bri > 0 else '어두울수록 비싼 경향'}")

# 보고서용 데이터 저장
sample_df[['title', 'lprice', 'brightness', 'saturation']].to_csv('image_analysis_result.csv', index=False)
