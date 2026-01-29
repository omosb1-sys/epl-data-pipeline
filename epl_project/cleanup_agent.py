"""
Antigravity Cleanup Agent
Mac(8GB RAM) 및 디스크 공간 최적화를 위한 자동 관리 시스템.
1. 7일 경과된 아카이브 로그 자동 삭제.
2. .agent 폴더 용량 10MB 초과 시 FIFO 삭제.
3. 핵심 지식 증류(Distillation) 가이드 제공.
"""

import os
import time
import json
from pathlib import Path

class CleanupAgent:
    def __init__(self):
        self.base_dir = Path("/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project")
        self.archive_dir = self.base_dir / ".agent" / "scratchpad" / "archive"
        self.agent_dir = self.base_dir / ".agent"
        self.memory_file = self.base_dir / "data" / "team_memory.json"
        self.max_quota_mb = 10
        self.ttl_days = 7

    def get_folder_size(self, folder):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    def run_cleanup(self):
        print("🧼 [Cleanup Agent] 리소스 최적화 및 청소 시작...")
        
        # 1. TTL 기반 삭제 (7일)
        now = time.time()
        if self.archive_dir.exists():
            for f in self.archive_dir.iterdir():
                if f.is_file() and now - f.stat().st_mtime > (self.ttl_days * 86400):
                    print(f"🗑️ [TTL] 7일 경과 파일 삭제: {f.name}")
                    f.unlink()

        # 2. Quota 기반 삭제 (10MB)
        current_size = self.get_folder_size(self.agent_dir)
        if current_size > (self.max_quota_mb * 1024 * 1024):
            print(f"⚠️ [Quota] 용량 초과 ({current_size / 1024 / 1024:.2f}MB). 오래된 파일부터 삭제합니다.")
            # 모든 아카이브 파일을 시간순으로 정렬
            files = sorted(
                [f for f in self.archive_dir.glob("**/*") if f.is_file()],
                key=lambda x: x.stat().st_mtime
            )
            for f in files:
                if self.get_folder_size(self.agent_dir) <= (self.max_quota_mb * 1024 * 1024):
                    break
                print(f"🗑️ [Quota] 용량 확보를 위해 삭제: {f.name}")
                f.unlink()

        print(f"✅ [Cleanup Agent] 청소 완료. 현재 .agent 용량: {self.get_folder_size(self.agent_dir) / 1024 / 1024:.2f}MB")

    def distill_knowledge(self, insight_key: str, content: str):
        """핵심 지식을 메모리에 영구 저장 (파일은 삭제될 수 있으므로)"""
        memory = {}
        if self.memory_file.exists():
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory = json.load(f)
        
        memory[insight_key] = {
            "content": content,
            "distilled_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=4)
        print(f"🧠 [Distillation] 핵심 지식 저장 완료: {insight_key}")

if __name__ == "__main__":
    agent = CleanupAgent()
    agent.run_cleanup()
