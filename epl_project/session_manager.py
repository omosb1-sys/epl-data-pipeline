import json
import os
from datetime import datetime
import pandas as pd

LOG_FILE = os.path.join(os.path.dirname(__file__), "internal", "session_history.jsonl")

def log_session_activity(activity_type: str, detail: str, tokens_estimate: int = 0):
    """
    Antigravity의 활동 내역을 기록합니다.
    """
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": activity_type,
        "detail": detail,
        "tokens": tokens_estimate
    }
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def get_session_stats():
    """
    저장된 기록을 바탕으로 통계를 리턴합니다.
    """
    if not os.path.exists(LOG_FILE):
        return None
    
    df = pd.read_json(LOG_FILE, lines=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

if __name__ == "__main__":
    # 초기화 및 테스트
    log_session_activity("INITIALIZE", "Antigravity Insight System Started")
    print(f"✅ Session logging initialized at {LOG_FILE}")
