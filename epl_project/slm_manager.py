"""
SLM Manager: 하드웨어 맞춤형 소형 모델 관리자
================================================
8GB RAM 맥 환경을 위한 초경량 모델 라우팅 및 최적화

주기능:
1. 하드웨어 감지: 시스템 메모리 확인 후 최적 모델 티어 결정
2. 모델 라우팅: 8GB RAM일 경우 1.5B~3B 모델 우선 배정
3. 가속 확인: Metal 가속 여부 체크 및 안내

Author: Antigravity (Hardware-Aware Architect)
Date: 2026-01-23
"""

import os
import psutil
import subprocess
import json

class SLMManager:
    """하드웨어 상태를 고려하여 최적의 SLM을 선택하고 관리합니다."""
    
    def __init__(self):
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.arch = subprocess.run(['uname', '-m'], capture_output=True, text=True).stdout.strip()
        self.is_intel = self.arch == 'x86_64'
        if os.uname().sysname == 'Darwin':
            try:
                self.cpu_brand = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
            except FileNotFoundError:
                self.cpu_brand = "Apple Silicon (Unknown)"
        else:
            self.cpu_brand = "Linux Generic Server"
        
    def get_optimal_model(self) -> str:
        """현재 시스템에 가장 적합한 모델명을 반환합니다."""
        if self.is_intel:
            # Intel Mac: GPU 가속이 약하므로 0.5B ~ 1.5B 초대형 모델 추천
            return "qwen2.5:0.5b" if self.total_ram_gb <= 8.5 else "qwen2.5:1.5b"
        
        if self.total_ram_gb <= 8.5:
            return "qwen2.5:1.5b"
        elif self.total_ram_gb <= 16.5:
            return "qwen2.5:7b"
        else:
            return "llama3.1:8b"

    def get_optimal_embedding_model(self) -> str:
        """[Unsloth SOTA] 하드웨어 성능과 Fine-tuning 효율을 고려한 임베딩 모델 추천"""
        if self.is_intel:
            # Intel Mac (8GB): Unsloth 최적화 베이스라인인 MiniLM 추천
            return "sentence-transformers/all-MiniLM-L6-v2"
        else:
            # Apple Silicon: Unsloth가 2배 이상 가속하는 ModernBERT 또는 BGE-M3 추천
            if self.total_ram_gb > 16:
                return "BAAI/bge-m3" # 고성능 다국어 임베딩
            return "answerdotai/ModernBERT-large" # 최신 SOTA (Speed 2.2x up)

    def check_ollama_status(self):
        """로컬 Ollama 및 모델 설치 상태를 점검합니다."""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode != 0:
                return False, "Ollama가 설치되어 있지 않거나 실행 중이 아닙니다."
            
            models = result.stdout
            optimal = self.get_optimal_model()
            
            if optimal in models:
                return True, f"최적의 모델({optimal})이 감지되었습니다."
            else:
                return False, f"최적 모델({optimal})이 없습니다. 'ollama pull {optimal}'이 필요합니다."
        except FileNotFoundError:
            return False, "Ollama 커맨드를 찾을 수 없습니다."

    def get_hardware_report(self):
        """사용자에게 전달할 하드웨어 최적화 리포트를 생성합니다."""
        advice = ""
        if self.is_intel:
            advice = (f"현재 {self.cpu_brand} (MacBook Air 2017) 환경입니다. "
                     "이 사양은 Dual-Core CPU와 DDR3 메모리를 사용하므로 0.5B 모델이 가장 적합합니다. "
                     "STEM 원칙에 따라, 대형 임베딩은 RAM에 오프로드하고 CPU 쓰레드 수를 2~3개로 제한하여 "
                     "발열과 병목 현상을 방지하는 하이브리드 추론 모드를 권장합니다.")
        else:
            advice = "Apple Silicon 환경입니다. Metal 가속을 통해 1.5B 모델을 매우 민첩하게 사용할 수 있습니다."

        report = {
            "Total RAM": f"{self.total_ram_gb:.1f} GB",
            "Architecture": self.arch,
            "Target Model": self.get_optimal_model(),
            "Advice": advice
        }
        return report

    def query(self, prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.7) -> str:
        """Ollama API를 통해 로컬 모델에게 질의합니다."""
        import requests
        
        model = self.get_optimal_model()
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"Error: API returned status {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == "__main__":
    manager = SLMManager()
    print("🖥️ 하드웨어 진단 결과:")
    print(json.dumps(manager.get_hardware_report(), indent=2, ensure_ascii=False))
    
    status, msg = manager.check_ollama_status()
    print(f"\n✅ 상태 점검: {msg}")
