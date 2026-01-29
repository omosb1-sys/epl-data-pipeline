import pandas as pd
import numpy as np
import glob
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 데이터 로드
data_dir = "./naver_shopping_analysis/data"
csv_files = glob.glob(os.path.join(data_dir, "naver_shopping_*.csv"))
df = pd.concat([pd.read_csv(f, encoding='utf-8-sig') for f in csv_files], ignore_index=True)

# 전처리
df = df.drop_duplicates(subset=['product_id']).copy()
df['lprice'] = pd.to_numeric(df['lprice'], errors='coerce')
df = df.dropna(subset=['lprice', 'title'])

# 이상치 제거 (너무 싼/비싼 것 제외 - 모델 안정성 위함)
# 하위 1%, 상위 1% 제거
lower = df['lprice'].quantile(0.01)
upper = df['lprice'].quantile(0.99)
df_clean = df[(df['lprice'] >= lower) & (df['lprice'] <= upper)].copy()

print(f"🧹 데이터 정제 후: {len(df_clean)}개 샘플")

# 2. 텍스트 전처리 & 벡터화 (TF-IDF)
# 한글/영문만 남기고 제거
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^가-힣a-z0-9\s]', ' ', text)
    return text

df_clean['clean_title'] = df_clean['title'].apply(clean_text)

# Word2Vec 대신 TF-IDF를 사용하는 전략적 이유:
# 사용자가 '스탠리' 같은 "어떤 단어"가 가격을 올리는지 알고 싶어함.
# 임베딩(Vector)은 해석이 어렵지만, TF-IDF + 회귀계수는 "단어별 가격 기여도"를 정확히 산출 가능.
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)
X = vectorizer.fit_transform(df_clean['clean_title'])
y = np.log1p(df_clean['lprice']) # 가격은 로그 변환 (정규분포 근사)

# 3. 모델 학습 (Ridge Regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 4. 평가
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_real = np.expm1(y_test)

mae = mean_absolute_error(y_real, y_pred)
r2 = r2_score(y_real, y_pred)

print(f"\n🚀 모델 성능 평가")
print(f"평균 오차(MAE): 약 {mae:,.0f}원")
print(f"설명력(R2 Score): {r2:.3f}")

# 5. 인사이트 추출: 가격을 올리는/내리는 핵심 키워드
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_

# 데이터프레임으로 변환
coef_df = pd.DataFrame({'keyword': feature_names, 'coefficient': coefs})
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values(by='coefficient', ascending=False)

print("\n💎 [Premium Keywords] 가격을 상승시키는 단어 TOP 15")
print(coef_df.head(15)[['keyword', 'coefficient']])

print("\n📉 [Budget Keywords] 가격을 하락시키는 단어 TOP 15")
print(coef_df.tail(15)[['keyword', 'coefficient']].sort_values(by='coefficient'))

# 6. 가격 예측기 함수 (데모)
def predict_price(title):
    clean = clean_text(title)
    vec = vectorizer.transform([clean])
    pred_log = model.predict(vec)
    pred_price = np.expm1(pred_log)[0]
    return pred_price

print("\n🧪 [가격 예측 시뮬레이션]")
test_titles = [
    "스타벅스 대용량 텀블러",
    "스탠리 퀜처 한정판",
    "다이소 가성비 물병"
]

for t in test_titles:
    price = predict_price(t)
    print(f"상품명: '{t}' --> 예상 가격: {price:,.0f}원")

# 7. 결과 저장 (시각화용)
coef_df.head(20).to_csv('top_positive_keywords.csv', index=False)
coef_df.tail(20).to_csv('top_negative_keywords.csv', index=False)
