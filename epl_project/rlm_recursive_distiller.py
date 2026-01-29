
"""
[Recursive Context Distiller v1.0]
Inspired by Claude Code RLM (Recursive Language Model)
======================================================
이 가이드는 Antigravity Rule 36(RLM Strategy)을 실무적으로 구현하여
방대한 코드베이스나 문서를 정보 손실 없이 요약/응축하는 방법을 제시합니다.
"""

import os
from pathlib import Path
from loguru import logger

def recursive_summarize(path: Path, depth: int = 0) -> str:
    """
    [Rule 36.1] Recursive Context Processing
    디렉토리 구조를 재귀적으로 돌며 하위 요약을 상위로 전파합니다.
    """
    indent = "  " * depth
    if path.is_file():
        # 파일의 경우 핵심 메타데이터와 시맨틱 요약 (여기서는 예시로 파일명만)
        logger.debug(f"{indent}📄 Processing File: {path.name}")
        return f"{path.name} (File: {path.suffix})"
    
    if path.is_dir():
        logger.info(f"{indent}📂 Summarizing Directory: {path.name}")
        sub_summaries = []
        for item in path.iterdir():
            if item.name.startswith('.') or item.name == '__pycache__':
                continue
            sub_summaries.append(recursive_summarize(item, depth + 1))
        
        # 하위 요소들의 요약을 하나로 응축 (Distillation)
        summary = f"Dir [{path.name}] contains: [" + ", ".join(sub_summaries) + "]"
        return summary

def rlm_distillation_workflow(project_root: str):
    """
    [Rule 36.4] Automated Distillation Engine
    전체 프로젝트 구조를 RLM 패턴으로 정제하여 'Global Context'를 생성합니다.
    """
    root = Path(project_root)
    logger.info(f"🌀 [RLM] '{root.name}' 프로젝트 재귀적 응축 시작...")
    
    final_context = recursive_summarize(root)
    
    # 이 요약본은 Gemini 3의 컨텍스트 창에 주입되거나 project_knowledge.json에 보관됨
    logger.success(f"✨ RLM Distillation 완료. 요약 밀도: {len(final_context)} chars")
    return final_context

if __name__ == "__main__":
    # 현재 epl_project 폴더를 대상으로 시뮬레이션
    target_path = Path(__file__).parent
    summary = rlm_distillation_workflow(str(target_path))
    
    print("\n--- Final Root Summary (Distilled Global Context) ---")
    print(summary[:500] + "..." if len(summary) > 500 else summary)
