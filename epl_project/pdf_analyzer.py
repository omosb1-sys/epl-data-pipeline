"""
PDF 문서 분석 도구
==================
AI 논문 PDF를 읽고 핵심 내용을 추출합니다.

Author: Antigravity AI
Date: 2026-01-22
"""

import PyPDF2
import re
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF에서 텍스트 추출"""
    text = ""
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        
        print(f"📄 PDF 정보:")
        print(f"   - 총 페이지: {total_pages}페이지")
        print(f"   - 파일 크기: {Path(pdf_path).stat().st_size / 1024:.1f} KB")
        
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
            
    return text


def analyze_pdf_structure(text: str):
    """PDF 구조 분석"""
    print("\n📊 문서 구조 분석:")
    
    # 총 글자 수
    total_chars = len(text)
    print(f"   - 총 글자 수: {total_chars:,}자")
    
    # 줄 수
    lines = text.split('\n')
    print(f"   - 총 줄 수: {len(lines):,}줄")
    
    # 키워드 빈도 분석
    keywords = ['AI', '인공지능', '머신러닝', '딥러닝', '학습', '모델', '데이터']
    print("\n🔍 주요 키워드 빈도:")
    for keyword in keywords:
        count = text.count(keyword)
        if count > 0:
            print(f"   - '{keyword}': {count}회")


def extract_sections(text: str):
    """섹션별로 내용 추출"""
    print("\n📑 섹션 추출:")
    
    # 제목 패턴 찾기 (숫자. 제목 형식)
    section_pattern = r'(\d+\.\s+[^\n]+)'
    sections = re.findall(section_pattern, text)
    
    if sections:
        print(f"   발견된 섹션: {len(sections)}개")
        for i, section in enumerate(sections[:10], 1):  # 처음 10개만
            print(f"   {i}. {section.strip()}")
    else:
        print("   섹션을 찾을 수 없습니다.")


def save_extracted_text(text: str, output_path: str):
    """추출된 텍스트 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\n💾 텍스트 저장: {output_path}")


def main():
    """메인 실행"""
    print("\n" + "📄" * 25)
    print("   AI 논문 PDF 분석 도구")
    print("📄" * 25 + "\n")
    
    pdf_path = "./data/pdfs/ai_논문.pdf"
    output_path = "./data/pdfs/ai_논문_extracted.txt"
    
    # PDF 텍스트 추출
    print("🔄 PDF 읽는 중...")
    text = extract_text_from_pdf(pdf_path)
    
    # 구조 분석
    analyze_pdf_structure(text)
    
    # 섹션 추출
    extract_sections(text)
    
    # 텍스트 저장
    save_extracted_text(text, output_path)
    
    # 미리보기
    print("\n📖 내용 미리보기 (처음 500자):")
    print("-" * 60)
    print(text[:500])
    print("-" * 60)
    
    print("\n✅ 분석 완료!")
    print(f"   전체 텍스트: {output_path}")


if __name__ == "__main__":
    main()
