import json
import os
import hashlib
import numpy as np
import scipy.stats as stats
from datetime import datetime

class ExperimentPlatform:
    """
    [Architect Mode] Enterprise-grade Experimentation Platform (Lightweight)
    Inspired by Uber & Spotify Engineering Blogs.
    """
    def __init__(self, log_path="epl_project/data/experiment_logs.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def get_user_bucket(self, user_id: str, experiment_id: str, salt: str = "epl_v1") -> str:
        """
        [Spotify Salt Machine Patent] 
        Deterministic hashing for consistent user experience.
        """
        hash_input = f"{user_id}:{experiment_id}:{salt}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        return "treatment" if (hash_val % 100) < 50 else "control"

    def log_assignment(self, user_id, experiment_id, bucket, metrics=None):
        """Unified logging for behavioral auditing."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "exp_id": experiment_id,
            "bucket": bucket,
            "metrics": metrics or {}
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def check_srm(self, control_count, treatment_count, expected_ratio=0.5):
        """
        [Statistical Rigor] Sample Ratio Mismatch (SRM) Check.
        Returns p-value. If p < 0.001, the experiment is biased.
        """
        total = control_count + treatment_count
        if total == 0: return 1.0
        
        expected_control = total * expected_ratio
        expected_treatment = total * (1 - expected_ratio)
        
        observed = [control_count, treatment_count]
        expected = [expected_control, expected_treatment]
        
        _, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
        return p_val

    def calculate_uplift(self, control_metric, treatment_metric):
        """Calculate percentage improvement."""
        if control_metric == 0: return 0.0
        return ((treatment_metric - control_metric) / control_metric) * 100

# Global singleton for the app
exp_platform = ExperimentPlatform()
