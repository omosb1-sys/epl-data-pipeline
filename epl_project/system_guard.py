import os
import time
import json
import subprocess
from datetime import datetime

class SystemGuard:
    """
    [Antigravity System Guard]
    Monitors machine resources (8GB RAM Mac) and project data bloat.
    Ensures the laptop stays fast by managing background processes and large files.
    """
    def __init__(self, threshold_mb=500, log_limit_mb=5):
        self.threshold_mb = threshold_mb
        self.log_limit_mb = log_limit_mb
        self.large_file_threshold = 20 * 1024 * 1024 # 20MB

    def inspect_system(self):
        """Returns a report of potential system bottlenecks."""
        bottlenecks = []
        
        # 1. Large Data Files Check
        for root, dirs, files in os.walk("."):
            if "node_modules" in root or ".git" in root: continue
            for f in files:
                fpath = os.path.join(root, f)
                try:
                    fsize = os.path.getsize(fpath)
                    if fsize > self.large_file_threshold:
                        if fpath.endswith(".csv"):
                            bottlenecks.append({
                                "type": "DATA_BLOAT",
                                "file": fpath,
                                "size_mb": round(fsize / (1024*1024), 2),
                                "recommendation": "Convert to Parquet and delete CSV."
                            })
                except: pass

        # 2. Memory Hog Processes Check
        try:
            # Check for multiple streamlit or python instances
            ps = subprocess.check_output(['ps', '-axo', 'pid,pmem,rss,comm'], text=True)
            relevant_ps = [line for line in ps.split('\n') if any(x in line for x in ['python', 'streamlit', 'node', 'ollama'])]
            if len(relevant_ps) > 5:
                bottlenecks.append({
                    "type": "PROCESS_BLOAT",
                    "count": len(relevant_ps),
                    "recommendation": "Close unused terminal tabs or app previews."
                })
        except: pass

        return bottlenecks

    def solve_common_problems(self):
        """Automatically clears cache and small temp files."""
        # Clear __pycache__
        subprocess.run(["find", ".", "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"])
        print("✅ Python cache cleared.")

# Singleton
system_guard = SystemGuard()

if __name__ == "__main__":
    report = system_guard.inspect_system()
    if not report:
        print("☀️  System is lean and healthy!")
    else:
        print("🚨 Potential Bottlenecks Found:")
        for item in report:
            print(f"- [{item['type']}] {item.get('file', 'Processes')}: {item.get('size_mb', '')}MB. {item['recommendation']}")
