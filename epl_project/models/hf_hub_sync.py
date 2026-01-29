from huggingface_hub import hf_hub_download
import os

class HFTacticHub:
    """
    [HF Tactic Hub] Hugging Face SOTA 모델 및 데이터 연동
    - Repository: google/timesfm-1.0-200m-pytorch (Alternative valid repo)
    - Protocol: Tier 1 Acquisition
    """
    def __init__(self, repo_id: str = "google/timesfm-1.0-200m-pytorch"):
        self.repo_id = repo_id
        self.local_dir = "epl_project/models/hf_assets"
        os.makedirs(self.local_dir, exist_ok=True)

    def fetch_model_config(self, filename: str = "config.json"):
        """SOTA 모델의 구성 파일을 HF에서 브링업"""
        try:
            print(f"📡 [HF Hub] Syncing {filename} from {self.repo_id}...")
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                local_dir=self.local_dir
            )
            print(f"✅ Success: {path}")
            return path
        except Exception as e:
            print(f"⚠️ [HF Hub] Download failed: {e}")
            return None

if __name__ == "__main__":
    hub = HFTacticHub()
    # 벤치마크용 설정 파일만 우선 확인
    hub.fetch_model_config()
