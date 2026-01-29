import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
pos_df = pd.read_csv('top_positive_keywords.csv')
neg_df = pd.read_csv('top_negative_keywords.csv').sort_values(by='coefficient')

# 시각화 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 가격 상승 키워드
plt.figure(figsize=(10, 8))
sns.barplot(data=pos_df.head(15), x='coefficient', y='keyword', palette='Reds_r')
plt.title('가격(Price)을 상승시키는 High-Value 키워드 TOP 15')
plt.xlabel('가격 예측 기여도 (Coefficient)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('insights_premium_keywords.png')

# 2. 가격 하락 키워드
plt.figure(figsize=(10, 8))
sns.barplot(data=neg_df.head(15), x='coefficient', y='keyword', palette='Blues_r')
plt.title('가격(Price)을 낮추는 Low-Value 키워드 TOP 15')
plt.xlabel('가격 예측 기여도 (Coefficient)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('insights_budget_keywords.png')

print("✅ 시각화 파일 생성 완료: insights_premium_keywords.png, insights_budget_keywords.png")
