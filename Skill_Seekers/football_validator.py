import pandas as pd
import numpy as np
from typing import List, Dict, Any

class FootballDataValidator:
    """
    K-League & EPL 전용 데이터 무결성 검증기 (Self-Created Skill)
    분석 전 데이터의 논리적 오류를 잡아내어 분석 신뢰도를 높입니다.
    """
    
    def __init__(self, data_type: str = "K-League"):
        self.data_type = data_type
        self.issues = []

    def validate_scores(self, df: pd.DataFrame) -> bool:
        """득점 데이터의 논리적 일관성 검증"""
        # 1. 음수 득점 확인
        if (df['home_score'] < 0).any() or (df['away_score'] < 0).any():
            self.issues.append("❌ 오류: 음수 득점 데이터가 발견되었습니다.")
            return False
        
        # 2. 결과와 점수의 일치성 확인 (간단하게)
        # (이 데이터셋 구조에 따라 추가 가능)
        return True

    def validate_duplicates(self, df: pd.DataFrame) -> bool:
        """중복 경기 데이터 확인"""
        dupes = df.duplicated(subset=['game_date', 'home_team_id', 'away_team_id']).sum()
        if dupes > 0:
            self.issues.append(f"⚠️ 경고: {dupes}개의 중복 경기 데이터가 존재합니다.")
            return False
        return True

    def run_full_diagnosis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """종합 진단 실행"""
        self.issues = []
        is_score_ok = self.validate_scores(df)
        is_dupe_ok = self.validate_duplicates(df)
        
        return {
            "is_valid": is_score_ok and is_dupe_ok,
            "issues": self.issues,
            "status": "Success" if is_score_ok and is_dupe_ok else "Warning/Error"
        }

if __name__ == "__main__":
    # 간단한 테스트
    test_df = pd.DataFrame({
        'game_date': ['2024-01-01', '2024-01-01'],
        'home_team_id': [1, 1],
        'away_team_id': [2, 2],
        'home_score': [1, 1],
        'away_score': [0, 0]
    })
    validator = FootballDataValidator()
    print(validator.run_diagnosis(test_df))
