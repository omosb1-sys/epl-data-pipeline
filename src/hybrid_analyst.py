"""
하이브리드 AI 분석 시스템
TinyLlama (로컬) + Gemini (클라우드) 최적 조합

GEMINI.md Protocol 준수
Hardware-Aware Hybrid Intelligence 전략
"""

import subprocess
import os
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup

try:
    from gemini_k_league_analyst import GeminiKLeagueAnalyst
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class HybridAnalyst:
    """TinyLlama + Gemini 하이브리드 분석 시스템"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            gemini_api_key: Gemini API 키 (선택사항)
        """
        self.tinyllama_available = self._check_tinyllama()
        
        if GEMINI_AVAILABLE and gemini_api_key:
            self.gemini = GeminiKLeagueAnalyst(api_key=gemini_api_key)
            self.gemini_available = True
        else:
            self.gemini = None
            self.gemini_available = False
    
    def _check_tinyllama(self) -> bool:
        """TinyLlama 설치 여부 확인"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "tinyllama" in result.stdout.lower()
        except Exception:
            return False
    
    def tinyllama_summarize(
        self, 
        text: str, 
        max_words: int = 100,
        timeout: int = 30
    ) -> Dict[str, str]:
        """
        TinyLlama로 영어 텍스트 요약 (빠름)
        
        Args:
            text: 요약할 영어 텍스트
            max_words: 최대 단어 수
            timeout: 타임아웃 (초)
            
        Returns:
            요약 결과 딕셔너리
        """
        if not self.tinyllama_available:
            return {"error": "TinyLlama가 설치되지 않았습니다."}
        
        prompt = f"""Summarize the following text in {max_words} words or less.
Focus on key facts and main points:

{text[:2000]}  # 처리 속도를 위해 2000자로 제한

Summary:"""
        
        try:
            start_time = datetime.now()
            
            result = subprocess.run(
                ["ollama", "run", "tinyllama", prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return {
                "summary": result.stdout.strip(),
                "elapsed_seconds": elapsed,
                "model": "tinyllama",
                "timestamp": datetime.now().isoformat()
            }
        except subprocess.TimeoutExpired:
            return {"error": f"타임아웃 ({timeout}초 초과)"}
        except Exception as e:
            return {"error": f"TinyLlama 실행 실패: {str(e)}"}
    
    def crawl_and_summarize(
        self, 
        url: str,
        translate_to_korean: bool = True
    ) -> Dict[str, str]:
        """
        영어 뉴스 크롤링 → TinyLlama 요약 → Gemini 한글 번역
        
        Args:
            url: 크롤링할 URL
            translate_to_korean: 한글 번역 여부
            
        Returns:
            분석 결과
        """
        # 1단계: 크롤링
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 본문 추출 (간단한 휴리스틱)
            article = soup.find('article')
            if article:
                text = article.get_text(separator=' ', strip=True)
            else:
                # article 태그 없으면 p 태그들 수집
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            if not text or len(text) < 100:
                return {"error": "유효한 본문을 찾을 수 없습니다."}
            
        except Exception as e:
            return {"error": f"크롤링 실패: {str(e)}"}
        
        # 2단계: TinyLlama 요약
        summary_result = self.tinyllama_summarize(text)
        
        if "error" in summary_result:
            return summary_result
        
        # 3단계: Gemini 한글 번역 (선택사항)
        if translate_to_korean and self.gemini_available:
            try:
                prompt = f"""
다음 영어 뉴스 요약을 한국어로 번역하고,
K-리그/EPL 팬들이 이해하기 쉽게 재구성하세요:

{summary_result['summary']}

간결하게 3~5문장으로 작성.
"""
                response = self.gemini.model.generate_content(prompt)
                korean_summary = response.text
                
                return {
                    "url": url,
                    "english_summary": summary_result['summary'],
                    "korean_summary": korean_summary,
                    "tinyllama_time": summary_result['elapsed_seconds'],
                    "total_time": summary_result['elapsed_seconds'] + 3,  # Gemini 약 3초
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "url": url,
                    "english_summary": summary_result['summary'],
                    "korean_summary": f"번역 실패: {str(e)}",
                    "tinyllama_time": summary_result['elapsed_seconds'],
                    "timestamp": datetime.now().isoformat()
                }
        else:
            return {
                "url": url,
                "english_summary": summary_result['summary'],
                "korean_summary": "(Gemini API 키 필요)",
                "tinyllama_time": summary_result['elapsed_seconds'],
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_summarize_news(
        self, 
        urls: List[str],
        save_report: bool = True
    ) -> Dict[str, any]:
        """
        여러 뉴스를 배치 요약 (비용 절감)
        
        Args:
            urls: URL 리스트
            save_report: 리포트 저장 여부
            
        Returns:
            배치 분석 결과
        """
        summaries = []
        total_time = 0
        
        for i, url in enumerate(urls):
            print(f"처리 중: {i+1}/{len(urls)} - {url}")
            result = self.crawl_and_summarize(url, translate_to_korean=False)
            
            if "error" not in result:
                summaries.append(result)
                total_time += result.get('tinyllama_time', 0)
        
        # Gemini로 최종 통합 리포트 생성 (1회만 호출)
        if self.gemini_available and summaries:
            combined = "\n\n".join([
                f"[{i+1}] {s['english_summary']}" 
                for i, s in enumerate(summaries)
            ])
            
            prompt = f"""
다음은 오늘의 EPL/K-리그 뉴스 {len(summaries)}개를 요약한 내용입니다.

{combined}

위 내용을 바탕으로 다음을 작성하세요:

## 🏆 오늘의 핵심 이슈 Top 3

각 이슈별로:
- **제목**: 한 문장 요약
- **내용**: 2~3문장 설명
- **의미**: 왜 중요한가?

한국어로 작성.
"""
            
            try:
                response = self.gemini.model.generate_content(prompt)
                final_report = response.text
            except Exception as e:
                final_report = f"최종 리포트 생성 실패: {str(e)}"
        else:
            final_report = "Gemini API 키가 필요합니다."
        
        result = {
            "total_articles": len(urls),
            "successful": len(summaries),
            "failed": len(urls) - len(summaries),
            "total_time_seconds": total_time,
            "average_time_per_article": total_time / len(summaries) if summaries else 0,
            "summaries": summaries,
            "final_report": final_report,
            "timestamp": datetime.now().isoformat()
        }
        
        # 리포트 저장
        if save_report:
            self._save_batch_report(result)
        
        return result
    
    def _save_batch_report(self, result: Dict) -> Path:
        """배치 리포트 저장"""
        output_dir = Path("../reports/batch")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_news_summary_{timestamp}.md"
        filepath = output_dir / filename
        
        content = f"""# 배치 뉴스 요약 리포트

**생성 시간:** {result['timestamp']}
**처리 기사:** {result['total_articles']}개
**성공:** {result['successful']}개
**실패:** {result['failed']}개
**총 소요 시간:** {result['total_time_seconds']:.2f}초
**평균 처리 시간:** {result['average_time_per_article']:.2f}초/기사

---

{result['final_report']}

---

## 📰 개별 기사 요약

"""
        
        for i, summary in enumerate(result['summaries']):
            content += f"""
### [{i+1}] {summary.get('url', 'N/A')}

**영어 요약:**
{summary.get('english_summary', 'N/A')}

**처리 시간:** {summary.get('tinyllama_time', 0):.2f}초

---
"""
        
        content += """
*Generated by Hybrid Analyst (TinyLlama + Gemini)*
*GEMINI.md Protocol v1.9*
"""
        
        filepath.write_text(content, encoding='utf-8')
        print(f"✅ 리포트 저장: {filepath}")
        return filepath
    
    def get_status(self) -> Dict[str, bool]:
        """시스템 상태 확인"""
        return {
            "tinyllama_available": self.tinyllama_available,
            "gemini_available": self.gemini_available,
            "hybrid_mode": self.tinyllama_available and self.gemini_available
        }


# 사용 예시
if __name__ == "__main__":
    # 초기화
    analyst = HybridAnalyst(gemini_api_key=os.getenv("GEMINI_API_KEY"))
    
    # 상태 확인
    status = analyst.get_status()
    print("시스템 상태:", json.dumps(status, indent=2))
    
    # 단일 뉴스 요약 테스트
    # url = "https://www.theguardian.com/football/premierleague"
    # result = analyst.crawl_and_summarize(url)
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 배치 뉴스 요약 테스트
    # urls = ["url1", "url2", "url3"]
    # batch_result = analyst.batch_summarize_news(urls)
    # print(batch_result['final_report'])
    
    print("✅ Hybrid Analyst 모듈 로드 완료")
