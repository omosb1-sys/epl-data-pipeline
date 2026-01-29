import time
import json
import os
from datetime import datetime

class AuditLogger:
    """
    [OpenAI-Inspired Observability]
    Tracks execution time, resource usage, and query complexity to ensure system scalability.
    """
    def __init__(self, log_path="logs/audit_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log_execution(self, agent_name: str, duration: float, query: str, success: bool = True):
        """Records the cost of an agent execution."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "duration_sec": round(duration, 4),
            "query_len": len(query),
            "status": "success" if success else "failed",
            "priority": "High" if "predict" in query.lower() or "analyze" in query.lower() else "Low"
        }
        
        # Keep log file size manageable (8GB RAM Mac environment)
        if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 2 * 1024 * 1024:
            self._rotate_logs()

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _rotate_logs(self):
        """OpenAI-style isolation: prunes old logs to keep the system stateless and lean."""
        if not os.path.exists(self.log_path): return
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Keep last 100 entries only
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.writelines(lines[-100:])

# Singleton instance
audit_logger = AuditLogger()
