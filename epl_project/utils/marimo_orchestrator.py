import os
import subprocess
import signal
import time
from pathlib import Path

class MarimoOrchestrator:
    def __init__(self):
        self.project_dir = Path("/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project")
        self.pid_file = self.project_dir / ".agent" / "marimo_session.pid"
        self.venv_path = self.project_dir / "models" / ".venv" / "bin" / "activate"
        # 기본 타겟을 현재 디렉토리로 설정하여 주피터 홈(파일 브라우저)처럼 보이게 함
        self.default_target = "."

    def is_running(self):
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                os.kill(pid, 0)
                return pid
            except (OSError, ValueError):
                self.pid_file.unlink()
        return None

    def start(self, target_file=None):
        """마리모 엔진 가동 (파일 지정 가능, 커널 에러 방지용 환경변수 추가)"""
        if self.is_running():
            print(f"✅ 마리모 엔진이 이미 실행 중입니다 (PID: {self.is_running()})")
            return

        target = target_file if target_file else self.default_target
        print(f"🚀 [Antigravity] 마리모 엔진 가동 중: {target} (주피터 홈 모드)...")

        # 커널 찾기 오류 방지를 위해 PYTHONPATH 및 VIRTUAL_ENV 명시
        venv_dir = self.project_dir / "models" / ".venv"
        env_setup = f"export VIRTUAL_ENV={venv_dir} && export PATH={venv_dir}/bin:$PATH && source {self.venv_path}"
        cmd = f"{env_setup} && marimo edit --headless --no-token --no-skew-protection {target}"
        
        log_file = self.project_dir / ".agent" / "marimo.log"
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                cmd,
                shell=True,
                executable="/bin/zsh",
                stdout=log,
                stderr=log,
                preexec_fn=os.setsid,
                cwd=str(self.project_dir)
            )
        
        self.pid_file.write_text(str(process.pid))
        url = "http://localhost:2718"
        print(f"✨ 엔진 가동 완료! 주소: {url}")
        
        # 중복 실행 방지: 이미 브라우저가 열려있을 가능성이 있으므로 
        # 짧은 대기 후 단 한 번만 'open' 명령을 수행하도록 제어합니다.
        # 또한 --headless 모드이므로 마리모 자체의 자동 실행은 억제된 상태입니다.
        try:
            time.sleep(1) # 서버 안정화 대기
            # 메모리 효율이 좋은 Comet 브라우저를 최우선으로 실행
            if os.path.exists("/Applications/Comet.app"):
                subprocess.run(["open", "-a", "Comet", url], check=False)
                print("🌐 Comet 브라우저를 통해 세션을 연결했습니다. (메모리 최적화 모드)")
            elif os.path.exists("/Applications/Google Chrome.app"):
                subprocess.run(["open", "-a", "Google Chrome", url], check=False)
                print("🌐 Google Chrome을 통해 세션을 연결했습니다.")
            else:
                subprocess.run(["open", url], check=False)
                print("🌐 기본 브라우저를 통해 세션을 연결했습니다.")
        except Exception:
            pass

    def stop(self):
        pid = self.is_running()
        if pid:
            print(f"🧹 [Antigravity] 리소스 회수 중 (마리모 종료)...")
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                self.pid_file.unlink()
                print("✅ 엔진 종료 및 RAM 자원 반환 완료.")
            except Exception as e:
                print(f"❌ 종료 중 오류 발생: {e}")

if __name__ == "__main__":
    import sys
    orchestrator = MarimoOrchestrator()
    if len(sys.argv) > 1:
        action = sys.argv[1]
        target = sys.argv[2] if len(sys.argv) > 2 else None
        if action == "start":
            orchestrator.start(target)
        elif action == "stop":
            orchestrator.stop()
        elif action == "restart":
            orchestrator.stop()
            time.sleep(1)
            orchestrator.start(target)
