import pandas as pd
import os
from typing import Optional

class StatsBombMaster:
    """
    StatsBomb Open Data 수급 및 분석 도구 (Self-Created Skill)
    Hugging Face와 GitHub의 축구 데이터를 직접 연결합니다.
    """
    
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
        print("⚔️ StatsBomb Master 스킬 활성화: 데이터 발견 모드")

    def download_sample_matches(self, competition_id: int = 11, season_id: int = 90) -> Optional[pd.DataFrame]:
        """특정 시즌의 경기 목록 가져오기 (예: La Liga 20/21)"""
        url = f"{self.base_url}matches/{competition_id}/{season_id}.json"
        try:
            df = pd.read_json(url)
            print(f"✅ 완료: {len(df)}개의 경기 데이터를 GitHub에서 확보했습니다.")
            return df
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None

    def get_event_data(self, match_id: int) -> Optional[pd.DataFrame]:
        """특정 경기의 상세 이벤트 데이터 확보"""
        url = f"{self.base_url}events/{match_id}.json"
        try:
            df = pd.read_json(url)
            return df
        except Exception as e:
            return None

if __name__ == "__main__":
    master = StatsBombMaster()
    master.download_sample_matches()
