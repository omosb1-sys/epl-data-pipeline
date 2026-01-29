from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# 1. 한글 폰트 등록 (Mac 전용 AppleGothic)
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
pdfmetrics.registerFont(TTFont('AppleGothic', FONT_PATH))

def create_ml_summary_pdf(output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # 2. 커스텀 스타일 정의
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontName='AppleGothic',
        fontSize=24,
        spaceAfter=30,
        alignment=1, # Center
        textColor=colors.HexColor("#2E4053")
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='AppleGothic',
        fontSize=16,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor("#1A5276"),
        borderPadding=5,
        underlineWidth=1
    )
    
    body_style = ParagraphStyle(
        'MainBody',
        parent=styles['Normal'],
        fontName='AppleGothic',
        fontSize=11,
        leading=16,
        spaceAfter=10
    )

    chapter_style = ParagraphStyle(
        'ChapterHeader',
        parent=styles['Heading3'],
        fontName='AppleGothic',
        fontSize=13,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#1E8449")
    )

    elements = []

    # 3. 문서 내용 구성
    # Title Page
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("머신러닝 필독서 핵심 요약 리포트", title_style))
    elements.append(Paragraph("<b>An Introduction to Statistical Learning (ISL)</b>", title_style))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("제시해주신 LinkedIn 링크의 머신러닝 교재 내용을 시니어 분석가의 시각으로 요약했습니다.", body_style))
    elements.append(Spacer(1, 40))

    # Introduction
    elements.append(Paragraph("1. 도서 개요", section_style))
    intro_txt = """
    본 도서는 머신러닝 및 통계학 입문자들에게 성경과도 같은 존재입니다. 복잡한 수학적 수식보다는 <b>기법의 직관적인 이해와 실전 적용</b>에 초점을 맞추고 있으며, 
    최근 전 세계적인 흐름에 발맞춰 기존 R 버전 외에 <b>Python 버전(ISLP)</b>도 출간되어 데이터 분석가와 엔지니어들에게 필수적인 리소스가 되었습니다.
    """
    elements.append(Paragraph(intro_txt, body_style))

    # Core Chapters
    elements.append(Paragraph("2. 주요 챕터별 핵심 내용", section_style))
    
    chapters = [
        ("회귀 (Regression)", "선형 모델을 통해 연속적인 값을 예측하는 기초. 단순 회귀에서 다중 회귀로의 확장과 모델 평가(p-value, R-squared)를 다룹니다."),
        ("분류 (Classification)", "로지스틱 회귀, LDA 등 범주형 변수를 예측하는 기법. 비즈니스에서 부도 예측, 고객 이탈 예측 등에 가장 많이 쓰이는 영역입니다."),
        ("재표본 추출 (Resampling Methods)", "교차 검증(Cross-validation)과 부트스트랩을 통해 모델의 성능을 정교하게 추정하고 과적합(Overfitting)을 방지합니다."),
        ("선형 모델 선택 및 정규화", "라쏘(Lasso)와 릿지(Ridge) 회귀를 통해 불필요한 변수를 제거하고 고차원 데이터에서 모델의 해석력을 높입니다."),
        ("트리 기반 방법 (Tree-based Methods)", "의사결정 나무, 랜덤 포레스트, 부스팅 등을 다루며 현대 머신러닝 경진대회(Kaggle 등)의 주력 알고리즘들을 설명합니다."),
        ("비지도 학습 (Unsupervised Learning)", "PCA(주성분 분석)와 군집화(Clustering)를 통해 사전 정보 없이 데이터 내부의 숨겨진 패턴과 그룹을 찾아냅니다."),
        ("딥러닝 및 최신 기법", "기본적인 신경망 구조부터 합성곱(CNN), 순환신경망(RNN) 및 현대적 딥러닝 프레임워크의 기초를 다룹니다.")
    ]

    for title, desc in chapters:
        elements.append(Paragraph(f"• <b>{title}</b>", chapter_style))
        elements.append(Paragraph(desc, body_style))

    # Learning Path Strategy
    elements.append(Paragraph("3. 시니어 분석가의 학습 전략 제언", section_style))
    strategy_txt = """
    이 책을 통해 단순한 코드 복붙(Copy-Paste)이 아닌, <b>알고리즘이 작동하는 기전</b>을 이해하는 것이 중요합니다. 
    특히 각 장 끝에 있는 <b>'Lab'</b> 세션을 통해 실무 데이터에 적용하는 감각을 익히시기 바랍니다. 
    본 프로젝트(Kmong)에서도 이 책에 나오는 회귀 분석과 트리 기반 모델들이 핵심적인 역할을 하게 될 것입니다.
    """
    elements.append(Paragraph(strategy_txt, body_style))

    # Footer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("<hr/>", body_style))
    elements.append(Paragraph("⏱️ 요약 생성 시간: 2026.01.23 | Powered by Antigravity AI Project", body_style))

    # PDF 빌드
    doc.build(elements)
    print(f"✅ 머신러닝 요약 PDF 생성 완료: {output_path}")

if __name__ == "__main__":
    output_file = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/ML_Book_Summary_ISL.pdf"
    create_ml_summary_pdf(output_file)
