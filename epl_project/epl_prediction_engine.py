import xgboost as xgb
import optuna
import pandas as pd
import numpy as np
from datetime import datetime
from epl_duckdb_manager import EPLDuckDBManager

class EPLPredictionEngine:
    def __init__(self):
        self.db = EPLDuckDBManager()
        self.model = None
        # 샘플 가중치 (실제로는 학습을 통해 얻어야 함)
        self.feature_weights = {
            'home_advantage': 0.1,
            'recent_form': 0.2,
            'squad_value': 0.4,
            'injury_impact': -0.1
        }

    def train_with_optuna(self, X_train, y_train):
        """
        LinkedIn 아이디어: Optuna를 이용한 XGBoost 하이퍼파라미터 튜닝
        (현재는 뼈대만 구축, 실제 데이터 유입 시 가동)
        """
        def objective(trial):
            param = {
                'verbosity': 0,
                'objective': 'multi:softprob',
                'num_class': 3,
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
            }
            # 실제 데이터가 있을 때 xgb.train 실행
            return 0.5 # Dummy accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        print(f"✅ Optuna Optimization Complete. Best Params: {study.best_params}")
        return study.best_params

    def predict_match(self, fixture_id: int):
        """
        특정 경기의 승률 예측 및 가치 베팅(Value Bet) 계산
        """
        # 1. 경기 정보 및 배당률 가져오기
        odds_df = self.db.get_latest_odds(fixture_id)
        if odds_df.empty:
            return None

        # 2. [Simulated Inference] XGBoost 모델 예측 (여기서는 규칙 기반 시뮬레이션)
        # 실제 운영시 self.model.predict() 사용
        home_win_prob = 0.45 
        draw_prob = 0.25
        away_win_prob = 0.30

        # 3. Value Bet 계산 (예측 확률 * 배당률 - 1)
        home_odds = odds_df['home_win_odds'].values[0]
        draw_odds = odds_df['draw_odds'].values[0]
        away_odds = odds_df['away_win_odds'].values[0]

        edges = {
            'Home': home_win_prob * home_odds - 1,
            'Draw': draw_prob * draw_odds - 1,
            'Away': away_win_prob * away_odds - 1
        }

        # 가장 큰 Edge를 가진 쪽 선택
        best_side = max(edges, key=edges.get)
        best_edge = edges[best_side]

        # 4. 결과 저장
        pred_df = pd.DataFrame([{
            "fixture_id": fixture_id,
            "timestamp": datetime.now(),
            "home_win_prob": home_win_prob,
            "draw_prob": draw_prob,
            "away_win_prob": away_win_prob,
            "value_bet_side": best_side if best_edge > 0.05 else "No Bet", # 5% 이상의 우위가 있을 때만
            "value_bet_edge": best_edge
        }])
        
        self.db.insert_prediction(pred_df)
        return pred_df

    def run_all_predictions(self):
        """모든 예정된 경기에 대해 예측 실행"""
        fixtures = self.db.conn.execute("SELECT fixture_id FROM fixtures WHERE status != 'Finished'").df()
        for fid in fixtures['fixture_id']:
            self.predict_match(fid)
        print("✅ All Predictions Updated.")

if __name__ == "__main__":
    engine = EPLPredictionEngine()
    engine.run_all_predictions()
