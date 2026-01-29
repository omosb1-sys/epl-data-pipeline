import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

class AntigravityOptimizer:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.skills_dir = self.project_root / ".agent" / "skills"
        self.memory_path = self.project_root / "data" / "team_memory.json"
        
    def disable_heavy_skills(self):
        """특정 분야(프론트엔드 등) 스킬 비활성화"""
        target_skills = [
            "vercel-best-practices",
            "web-design-guidelines",
            "insight-manager"
        ]
        count = 0
        for skill in target_skills:
            src = self.skills_dir / skill
            dst = self.skills_dir / f"{skill}.disabled"
            if src.exists():
                os.rename(src, dst)
                count += 1
        return count

    def kill_duplicate_helpers(self):
        """중복된 헬퍼 프로세스 정리"""
        try:
            # macOS 기준 Antigravity Helper 프로세스 종료
            subprocess.run(["pkill", "-f", "Antigravity Helper (Plugin)"], capture_output=True)
            return True
        except Exception:
            return False

    def compress_context(self):
        """로그 정리 및 컨텍스트 압축"""
        # 기존 cleanup_agent.py 호출 (있을 경우)
        cleanup_script = self.project_root / "cleanup_agent.py"
        if cleanup_script.exists():
            subprocess.run(["python3", str(cleanup_script)], capture_output=True)
        return True

    def record_memory(self):
        """최적화 이력 기록"""
        if not self.memory_path.exists():
            data = {}
        else:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data["Last_Optimization"] = f"[{timestamp}] 자동화 엔진에 의한 시스템 정밀 최적화 완료 (8GB RAM 모드)"
        
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def run_all(self):
        print(f"🚀 [Antigravity Optimizer] 작업을 시작합니다... ({datetime.now()})")
        s_count = self.disable_heavy_skills()
        print(f"📦 확장 기능 정리 완료: {s_count}개 비활성화")
        
        self.kill_duplicate_helpers()
        print("🗡️ 중복 프로세스 정리 완료")
        
        self.compress_context()
        print("🧹 컨텍스트 및 로그 압축 완료")
        
        self.record_memory()
        print("🧠 최적화 이력 장기 기억 저장 완료")
        print("✨ 시스템이 쾌적해졌습니다!")

if __name__ == "__main__":
    optimizer = AntigravityOptimizer()
    optimizer.run_all()
