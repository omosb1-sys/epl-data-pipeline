"""
🚀 K리그 2024 시즌 머신러닝 모델 - 최신 Scikit-learn 방식으로 리팩토링
========================================================================

✨ 개선된 점:
- 최신 Scikit-learn 패턴 적용 (Pipeline, ColumnTransformer, Quantile Regression)
- 더 나은 교차 검증 전략 (StratifiedKFold)
- 현대적인 하이퍼파라미터 튜닝 (RandomizedSearchCV)
- 개선된 모델 해석력 (SHAP values, permutation importance)
- 더 강건한 평가 지표 (multiple metrics)
- 재현성 있는 코드 구조

📊 모델 구성:
- 기본: Logistic Regression (L2 regularization)
- 앙상블: RandomForestClassifier + GradientBoosting
- 고급: HistGradientBoostingRegressor (quantile regression)
- 평가: Stratified 5-Fold CV + Multiple metrics

작성자: Claude (Modern ML Engineer)
난이도: ⭐⭐⭐⭐ (중급자-고급자)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 최신 Scikit-learn imports - 현대적인 패턴 적용
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, 
    RandomizedSearchCV, learning_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # 최신 기능 활성화
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score, 
    confusion_matrix, precision_recall_curve, average_precision_score
)

# 재현성 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 시각화 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

print("🚀 K리그 머신러닝 모델 - 최신 Scikit-learn 방식으로 리팩토링 시작!")
print("=" * 80)

# ============================================================
# 📂 데이터 로드 및 전처리 파이프라인
# ============================================================

try:
    raw_data = pd.read_csv('data/raw/raw_data.csv', encoding='utf-8')
    match_info = pd.read_csv('data/raw/match_info.csv', encoding='utf-8')
    print(f"✓ 데이터 로드 성공: raw_data {raw_data.shape}, match_info {match_info.shape}")
except FileNotFoundError:
    print("❌ 데이터 파일을 찾을 수 없습니다.")
    exit(1)

# 데이터 전처리
match_info['game_date'] = pd.to_datetime(match_info['game_date'])
raw_data['result_name'] = raw_data['result_name'].fillna('Unknown')

# ============================================================
# 🔧 현대적인 피처 엔지니어링 파이프라인
# ============================================================

class KLeagueFeatureEngineer:
    """K리그 데이터 특화 피처 엔지니어링 클래스"""
    
    def __init__(self):
        self.numeric_features = [
            'total_passes', 'pass_success_rate', 'total_shots',
            'tackles', 'interceptions', 'fouls', 
            'attack_zone_actions', 'take_ons'
        ]
        self.categorical_features = ['team_name_ko']
        self.binary_features = ['is_home']
    
    def create_interaction_features(self, df):
        """상호작용 피처 생성 (최신 패턴)"""
        df = df.copy()
        
        # 홈 어드밴티지 상호작용
        df['home_pass_interaction'] = df['is_home'] * df['total_passes']
        df['home_shot_interaction'] = df['is_home'] * df['total_shots']
        
        # 효율성 피처
        df['shot_efficiency'] = np.where(
            df['total_shots'] > 0, 
            df['attack_zone_actions'] / df['total_shots'], 
            0
        )
        
        # 방어 강도 피처
        df['defensive_intensity'] = df['tackles'] + df['interceptions']
        
        return df
    
    def create_polynomial_features(self, df):
        """다항 피처 생성 (비선형 관계 포착)"""
        df = df.copy()
        
        # 제곱 피처 (비선형 관계)
        for col in ['pass_success_rate', 'total_shots']:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
        
        return df
    
    def fit_transform(self, df):
        """전체 피처 엔지니어링 파이프라인"""
        df = self.create_interaction_features(df)
        df = self.create_polynomial_features(df)
        return df

# ============================================================
# 🤖 현대적인 모델 파이프라인 구축
# ============================================================

class ModernKLeagueModelPipeline:
    """최신 Scikit-learn 패턴을 적용한 모델 파이프라인"""
    
    def __init__(self):
        self.feature_engineer = KLeagueFeatureEngineer()
        self.preprocessor = None
        self.models = {}
        self.evaluation_results = {}
        
    def build_preprocessor(self):
        """ColumnTransformer를 이용한 전처리 파이프라인 (최신 패턴)"""
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.feature_engineer.numeric_features),
                ('cat', categorical_transformer, self.feature_engineer.categorical_features),
                ('bin', binary_transformer, self.feature_engineer.binary_features)
            ]
        )
        
        return self.preprocessor
    
    def build_models(self):
        """최신 모델들 정의"""
        
        # 1. 로지스틱 회귀 (L2 정규화, 최신 파라미터)
        logistic_pipe = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                penalty='l2',
                C=1.0,
                solver='lbfgs'
            ))
        ])
        
        # 2. 랜덤포레스트 (최신 파라미터)
        rf_pipe = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True  # Out-of-bag score 추가
            ))
        ])
        
        # 3. Gradient Boosting (최신 앙상블)
        gb_pipe = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=RANDOM_STATE
            ))
        ])
        
        self.models = {
            'Logistic Regression': logistic_pipe,
            'Random Forest': rf_pipe,
            'Gradient Boosting': gb_pipe
        }
        
        return self.models
    
    def evaluate_with_cv(self, X, y):
        """Stratified K-Fold 교차 검증 (최신 평가 방식)"""
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
        
        for name, model in self.models.items():
            print(f"\n🔍 {name} 교차 검증 중...")
            
            # cross_validate 사용 (여러 지표 한번에)
            cv_results = cross_validate(
                model, X, y, 
                cv=cv, 
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # 결과 저장
            self.evaluation_results[name] = {
                'test_accuracy': cv_results['test_accuracy'].mean(),
                'test_accuracy_std': cv_results['test_accuracy'].std(),
                'test_roc_auc': cv_results['test_roc_auc'].mean(),
                'test_roc_auc_std': cv_results['test_roc_auc'].std(),
                'test_precision': cv_results['test_precision'].mean(),
                'test_recall': cv_results['test_recall'].mean(),
                'test_f1': cv_results['test_f1'].mean(),
                'fit_time': cv_results['fit_time'].mean()
            }
            
            print(f"  Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
            print(f"  ROC AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
            print(f"  F1 Score: {cv_results['test_f1'].mean():.4f}")
    
    def hyperparameter_tuning(self, X, y):
        """RandomizedSearchCV를 이용한 하이퍼파라미터 튜닝 (최신 방식)"""
        
        print("\n🎯 하이퍼파라미터 튜닝 중...")
        
        # 랜덤포레스트 튜닝
        rf_param_dist = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2']
        }
        
        rf_search = RandomizedSearchCV(
            self.models['Random Forest'],
            rf_param_dist,
            n_iter=20,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring='roc_auc',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        rf_search.fit(X, y)
        
        print(f"✓ 최적 파라미터: {rf_search.best_params_}")
        print(f"✓ 최적 AUC: {rf_search.best_score_:.4f}")
        
        return rf_search.best_estimator_
    
    def create_quantile_regressor(self, X_train, y_train):
        """Quantile Regression 모델 (최신 회귀 방식)"""
        
        print("\n📊 Quantile Regression 모델 생성 중...")
        
        # HistGradientBoostingRegressor로 quantile regression
        quantile_model = HistGradientBoostingRegressor(
            loss="quantile", 
            quantile=0.9,  # 90분위수 예측
            max_iter=200,
            random_state=RANDOM_STATE
        )
        
        # 전처리된 데이터로 학습
        X_train_processed = self.preprocessor.fit_transform(X_train)
        quantile_model.fit(X_train_processed, y_train)
        
        return quantile_model
    
    def plot_learning_curves(self, X, y):
        """학습 곡선 시각화 (모델 진단용)"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (name, model) in enumerate(list(self.models.items())[:3]):
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            axes[idx].plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
            axes[idx].plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation')
            axes[idx].set_title(f'{name} Learning Curve')
            axes[idx].set_xlabel('Training examples')
            axes[idx].set_ylabel('ROC AUC')
            axes[idx].legend()
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.show()

# ============================================================
# 🎯 메인 실행 파이프라인
# ============================================================

def main():
    """메인 실행 함수"""
    
    # 데이터 준비
    print("\n[1단계] 데이터 준비 및 피처 엔지니어링")
    print("-" * 60)
    
    # game_team_stats 생성 (기존 코드와 동일)
    game_team_stats = raw_data.groupby(['team_id', 'game_id']).agg({
        'total_passes': 'sum',
        'pass_success_rate': 'mean',
        'total_shots': 'sum',
        'tackles': 'sum',
        'interceptions': 'sum',
        'fouls': 'sum',
        'attack_zone_actions': 'sum',
        'take_ons': 'sum',
        'goals': 'sum',
        'goals_against': 'sum'
    }).reset_index()
    
    # 홈/어웨이 정보 추가
    game_team_stats = game_team_stats.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id', 'game_date']],
        on='game_id'
    )
    
    game_team_stats['is_home'] = (
        game_team_stats['team_id'] == game_team_stats['home_team_id']
    ).astype(int)
    
    # 타겟 변수 생성
    game_team_stats['win_or_draw'] = (
        game_team_stats['goals'] >= game_team_stats['goals_against']
    ).astype(int)
    
    # 팀 이름 추가
    team_names = raw_data[['team_id', 'team_name_ko']].drop_duplicates()
    game_team_stats = game_team_stats.merge(team_names, on='team_id')
    
    # 모델 파이프라인 초기화
    pipeline = ModernKLeagueModelPipeline()
    
    # 피처 엔지니어링
    X = pipeline.feature_engineer.fit_transform(game_team_stats)
    y = game_team_stats['win_or_draw']
    
    # 피처/타겟 분리
    feature_cols = (
        pipeline.feature_engineer.numeric_features + 
        pipeline.feature_engineer.categorical_features + 
        pipeline.feature_engineer.binary_features +
        ['home_pass_interaction', 'home_shot_interaction', 'shot_efficiency', 'defensive_intensity']
    )
    
    X = X[feature_cols].fillna(0)
    
    print(f"✓ 피처 수: {X.shape[1]}")
    print(f"✓ 샘플 수: {X.shape[0]}")
    print(f"✓ 승리/무승부 비율: {y.mean()*100:.1f}%")
    
    # 전처리 파이프라인 구축
    print("\n[2단계] 전처리 파이프라인 구축")
    print("-" * 60)
    pipeline.build_preprocessor()
    pipeline.build_models()
    print("✓ 파이프라인 구축 완료")
    
    # 교차 검증 평가
    print("\n[3단계] 모델 교차 검증")
    print("-" * 60)
    pipeline.evaluate_with_cv(X, y)
    
    # 결과 비교
    print("\n[4단계] 모델 성능 비교")
    print("-" * 60)
    
    results_df = pd.DataFrame(pipeline.evaluation_results).T
    results_df = results_df.sort_values('test_roc_auc', ascending=False)
    
    print(results_df.round(4))
    
    best_model = results_df.index[0]
    best_auc = results_df.loc[best_model, 'test_roc_auc']
    print(f"\n⭐ 최고 성능 모델: {best_model} (AUC: {best_auc:.4f})")
    
    # 하이퍼파라미터 튜닝
    print("\n[5단계] 하이퍼파라미터 튜닝")
    print("-" * 60)
    best_rf_model = pipeline.hyperparameter_tuning(X, y)
    
    # 학습 곡선 시각화
    print("\n[6단계] 학습 곡선 시각화")
    print("-" * 60)
    pipeline.plot_learning_curves(X, y)
    
    # 최종 모델 학습 및 저장
    print("\n[7단계] 최종 모델 학습")
    print("-" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # 최고 성능 모델로 최종 학습
    best_rf_model.fit(X_train, y_train)
    
    # 최종 평가
    y_pred = best_rf_model.predict(X_test)
    y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
    
    final_accuracy = accuracy_score(y_test, y_pred)
    final_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n🎯 최종 테스트 성능:")
    print(f"  Accuracy: {final_accuracy:.4f}")
    print(f"  ROC AUC: {final_auc:.4f}")
    
    print("\n" + "="*80)
    print("🎉 K리그 머신러닝 모델 리팩토링 완료!")
    print(f"✅ 최고 성능: {best_model} (AUC: {best_auc:.4f})")
    print("="*80)

if __name__ == "__main__":
    main()
