import json
import os
from datetime import datetime

class DistillationEngine:
    """
    [PCL-Reasoner: Trace Distillation]
    Collects high-quality (Verified) reasoning traces for local SLM fine-tuning.
    Optimized for Unsloth-compatible JSONL format.
    """
    def __init__(self, trace_path="data/distilled_reasoning_traces.jsonl"):
        self.trace_path = trace_path
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)

    def save_verified_trace(self, query: str, reasoning_path: str, verifier_output: dict):
        """[Storage Guard] Saves trace with strict file size and resource management."""
        logic_score = verifier_output.get("logic_score", 0)
        if logic_score < 0.8: return

        # 1. 파일 크기 체크 (8GB RAM Mac 배려: 최대 5MB로 제한)
        # 5MB는 텍스트 데이터로서 수천 건의 고품질 추론을 담기에 충분한 공간입니다.
        if os.path.exists(self.trace_path) and os.path.getsize(self.trace_path) > 5 * 1024 * 1024:
            print("⚠️ [Storage Guard] Dataset limit reached. Rotating old traces...")
            self._rotate_logs()

        entry = {
            "instruction": "EPL 데이터 기반 심층 분석 및 추론을 수행하라.",
            "input": query,
            "output": reasoning_path,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "logic_score": logic_score
            }
        }
        
        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error: {e}")

    def _rotate_logs(self):
        """가장 오래된 20%의 데이터를 삭제하여 최신성을 유지하고 용량을 확보함."""
        if not os.path.exists(self.trace_path): return
        with open(self.trace_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 20% 삭제
        new_lines = lines[int(len(lines) * 0.2):]
        with open(self.trace_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    def get_collection_count(self) -> int:
        """Returns the number of verified traces collected so far."""
        if not os.path.exists(self.trace_path):
            return 0
        try:
            with open(self.trace_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except:
            return 0

# Singleton instance
distillation_engine = DistillationEngine()
