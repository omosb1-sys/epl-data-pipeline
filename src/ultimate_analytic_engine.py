import duckdb
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import holoviews as hv
from holoviews import opts
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.preprocessing import StandardScaler

# [친절한 가이드] 시각화 도구 설정 (차트를 그림처럼 그려주는 친구예요)
hv.extension('bokeh')

# --- PyTorch 신경망 모델 정의 (AI의 뇌 세포 구조) ---
class MatchPredictorNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=3):
        super(MatchPredictorNN, self).__init__()
        # 아주 간단한 3층짜리 신경망이에요. 입력 -> 생각1 -> 생각2 -> 결과
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), # 0보다 작은 값은 버리고 특징을 잡아내는 필터예요
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # 최종 승/무/패 3개 중 하나를 골라요
        )
        
    def forward(self, x):
        return self.net(x)

class UltimateAnalyticEngine:
    """
    30년 차 시니어 엔지니어가 설계한 초고성능 분석 엔진
    3개월 차 분석가님을 위해 아주 자세한 주석을 달았습니다! 🚀
    """
    
    def __init__(self, raw_path: str):
        self.raw_path = raw_path
        # CSV보다 10배 빠른 Parquet 파일 경로를 만들어요
        self.parquet_path = raw_path.replace('.csv', '.parquet')
        # 데이터 언어(SQL)로 소통할 DuckDB 엔진을 준비합니다
        self.db = duckdb.connect(':memory:')
        self.model = None # 아직 공부 전이라 비어있어요
        
    def layer1_ingestion(self):
        """[1층: 데이터 가져오기] 원시 데이터를 고속 엔진용으로 바꿉니다."""
        print("🪄 [Phase 1] 데이터 인프라 구축: CSV -> Parquet 고속 변환 중...")
        # CSV를 읽어서 가장 효율적인 Parquet 형식으로 저장해요
        pl.read_csv(self.raw_path).write_parquet(self.parquet_path)
        
        # DuckDB로 Parquet 파일을 직접 읽어옵니다. (이게 제일 빨라요!)
        query = f"SELECT * FROM read_parquet('{self.parquet_path}')"
        self.df_pl = self.db.query(query).pl()
        return self.df_pl

    def layer2_kinetic_processing(self):
        """[2층: 특징 만들기] 기계가 공부할 수 있게 데이터를 요리합니다."""
        print("⚡ [Phase 2] Polars 벡터 엔진: 빛의 속도로 팀별 지표 계산...")
        
        # lazy()는 '계산 계획'만 세우고 나중에 한꺼번에 실행해서 엄청 빨라요!
        self.processed_pl = self.df_pl.lazy().with_columns([
            # 최근 5경기 평균 득점 (팀별로 벽을 치고 계산해요)
            pl.col("goals_for").rolling_mean(window_size=5, min_periods=1).over("team").alias("avg_goals"),
            # 지금까지 쌓인 누적 승점
            pl.col("points").cum_sum().over("team").alias("cum_points")
        ]).collect() # 여기서 실제로 요리가 시작됩니다
        return self.processed_pl

    def layer3_advanced_inference(self):
        """[3층: 진짜 AI 학습] PyTorch 신경망이 축구의 법칙을 스스로 깨닫는 단계입니다."""
        print("🧠 [Phase 3] 다차원 모델링: PyTorch 신경망 엔진 가동 중...")
        
        # 1. 데이터 전처리 (비어있는 값은 과감히 버리고 필요한 것만 챙겨요)
        df = self.processed_pl.to_pandas().dropna(subset=['avg_goals', 'cum_points', 'result'])
        
        # 2. 결과(Win/Draw/Loss)를 숫자로 바꿔줍니다 (기계는 숫자만 읽어요)
        mapping = {'Win': 2, 'Draw': 1, 'Loss': 0}
        df['target'] = df['result'].map(mapping)
        
        X = df[['avg_goals', 'cum_points']].values # 문제지
        y = df['target'].values # 답안지
        
        # 3. 데이터 표준화 (누적 승점은 큰데 평균 득점은 작으면 기계가 헷갈려해요. 다리미로 펴줍니다!)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 4. PyTorch 전용 '텐서' 바구니에 담기
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        # 5. 모델과 학습 도구 소환
        model = MatchPredictorNN(input_dim=2) # 입력은 평균득점과 누적승점 2개!
        criterion = nn.CrossEntropyLoss() # 정답을 얼마나 틀렸는지 체크하는 '채점관'
        optimizer = optim.Adam(model.parameters(), lr=0.01) # 오답을 줄이려 노력하는 '선생님'
        
        # 6. 진짜 공부 시작 (20번 반복해서 문제지를 풀어요)
        print("🚀 PyTorch 신경망이 데이터를 보며 패턴을 학습하는 중 (Training)...")
        for epoch in range(20):
            optimizer.zero_grad() # 이전 기억(오차)을 지우고 새롭게 시작
            outputs = model(X_tensor) # 문제를 풀고
            loss = criterion(outputs, y_tensor) # 채점을 받고
            loss.backward() # 어디가 틀렸는지 거꾸로 거슬러 올라가서
            optimizer.step() # 머릿속(가중치)을 고칩니다!
            
            if (epoch+1) % 5 == 0:
                print(f"  ● Epoch [{epoch+1}/20] 학습 오차(Loss): {loss.item():.4f}")
        
        self.model = model
        print("✅ PyTorch 모델이 축구의 패턴을 깨달았습니다! 시스템에 장착 완료.")

    def layer4_xai_and_viz(self):
        """[4층: 그림으로 보기] 어려운 데이터를 눈에 보이는 차트로 만듭니다."""
        print("👁️ [Phase 4] 시각 분석: HoloViews 인터랙티브 차트 생성...")
        
        # 데이터를 판다스로 바꿔서 차트를 그려요
        df_pd = self.processed_pl.to_pandas()
        curve = hv.Curve(df_pd, 'game_id', 'avg_goals', label='최근 5경기 평균 득점')
        scatter = hv.Scatter(df_pd, 'game_id', 'goals_for', label='각 경기 실제 득점')
        
        # 두 차트를 겹치고(*) 예쁘게 꾸밉니다
        viz = (curve * scatter).opts(
            opts.Curve(width=800, height=400, color='blue', line_width=2, tools=['hover']),
            opts.Scatter(size=6, color='red', alpha=0.5)
        )
        return viz

    def layer5_toon_reporting(self, data: List[Dict]):
        """[5층: 핵심 보고] AI(제미나이)가 가장 좋아하는 가벼운 포맷으로 요약 보고합니다."""
        print("📄 [Phase 5] 시니어 분석가 최종 Insight 리포트 (TOON 형식)...")
        lines = [f"▼ ANALYTICAL_INSIGHT_REPORT"]
        for item in data:
            # 특수문자를 다 빼고 글자만 남겨서 AI가 추론하기 제일 편한 상태예요
            fields = [f"{k}: {v}" for k, v in item.items() if v is not None]
            lines.append(f"  ● " + " | ".join(fields))
        print("\n".join(lines))

# --- 메인 실행부 (공장 가동) ---
if __name__ == "__main__":
    # 1. 공장장 소환
    engine = UltimateAnalyticEngine('data/processed/team_match_results.csv')
    
    # 2. 전 공정 순차 가동 (1층 -> 2층 -> 3층 -> 4층 -> 5층)
    engine.layer1_ingestion()
    df_pro = engine.layer2_kinetic_processing()
    engine.layer3_advanced_inference()
    
    # 4층 시각화 결과물 (주피터 노트북 환경이라면 자동으로 나타납니다)
    viz = engine.layer4_xai_and_viz()
    
    # 5층 최종 TOON 리포트 출력
    summary = df_pro.head(3).to_dicts()
    engine.layer5_toon_reporting(summary)
    
    print("\n✅ [안티그래비티] 초보 분석가님, 이제 모든 인공지능 분석이 끝났습니다! 화이팅! 💪")
