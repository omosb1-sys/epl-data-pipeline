import json
import os

def run_deterministic_judge(response_text, required_keywords):
    """
    [B] LLM 판사보다 강력한 코드 기반 판사 (Deterministic Judge)
    Anthropic 가이드: 정규식이나 키워드 매칭이 모델 기반 평가보다 더 신뢰할 수 있음.
    """
    score = 100
    missing = []
    
    for kw in required_keywords:
        if kw not in response_text:
            score -= 15
            missing.append(kw)
            
    return score, missing

def run_regression_test(predict_func):
    """
    [A] 골든 세트 기반 회귀 테스트 (Regression Test)
    Anthropic 가이드: 20-50개의 고품질 예제만으로도 성능을 크게 조정 가능.
    """
    golden_path = os.path.join(os.path.dirname(__file__), "golden_set.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
        
    results = []
    print(f"🚀 에이전트 품질 검증 시작 (Golden Set Size: {len(golden_set)})")
    
    for case in golden_set:
        # 실제 앱의 예측 함수를 시뮬레이션
        # predict_func은 (home, away)를 받아 (prob, report_text)를 반환한다고 가정
        prob, report_text = predict_func(case['home'], case['away'])
        
        # 1. 수치 검증
        pass_prob = prob >= case['min_prob'] if case['expected_winner'] == case['home'] else (100-prob) >= case['min_prob']
        
        # 2. 내용 검증 (Code Judge)
        content_score, missing_kw = run_deterministic_judge(report_text, case['required_keywords'])
        
        results.append({
            "match": f"{case['home']} vs {case['away']}",
            "passed_prob": pass_prob,
            "content_score": content_score,
            "missing_keywords": missing_kw
        })
        
    return results

if __name__ == "__main__":
    # 내부 테스트용 모크 함수
    def mock_predict(h, a):
        return 85.0, "이 경기는 xG 데이터와 ELO 수치를 볼 때 승리 확률이 높습니다."
        
    test_results = run_regression_test(mock_predict)
    print(json.dumps(test_results, indent=2, ensure_ascii=False))
