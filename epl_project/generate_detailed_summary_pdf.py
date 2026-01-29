from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, HRFlowable
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# 1. 한글 폰트 등록 (Mac 전용 AppleGothic)
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
pdfmetrics.registerFont(TTFont('AppleGothic', FONT_PATH))

def create_detailed_ml_pdf(output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # 2. 커스텀 스타일 정의
    main_title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontName='AppleGothic',
        fontSize=26,
        spaceAfter=40,
        alignment=1, # Center
        textColor=colors.HexColor("#1B2631")
    )
    
    sub_title_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='AppleGothic',
        fontSize=14,
        alignment=1,
        spaceAfter=60,
        textColor=colors.HexColor("#5D6D7E")
    )

    part_style = ParagraphStyle(
        'PartHeading',
        parent=styles['Heading2'],
        fontName='AppleGothic',
        fontSize=18,
        spaceBefore=30,
        spaceAfter=15,
        textColor=colors.HexColor("#1A5276"),
        borderPadding=5
    )
    
    chapter_style = ParagraphStyle(
        'ChapterHeading',
        parent=styles['Heading3'],
        fontName='AppleGothic',
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor("#1E8449")
    )
    
    body_style = ParagraphStyle(
        'DetailedBody',
        parent=styles['Normal'],
        fontName='AppleGothic',
        fontSize=11,
        leading=18,
        spaceAfter=12,
        alignment=4 # Justify
    )

    quote_style = ParagraphStyle(
        'SeniorQuote',
        parent=styles['Normal'],
        fontName='AppleGothic',
        fontSize=12,
        leading=20,
        leftIndent=20,
        rightIndent=20,
        spaceBefore=30,
        textColor=colors.HexColor("#7D3C98"),
        backColor=colors.HexColor("#F4ECF7"),
        borderPadding=10
    )

    elements = []

    # --- Title Page ---
    elements.append(Spacer(1, 100))
    elements.append(Paragraph("머신러닝 바이블 집대성 리포트", main_title_style))
    elements.append(Paragraph("<b>An Introduction to Statistical Learning (ISL)</b>", main_title_style))
    elements.append(Paragraph("시니어 분석가의 시각으로 재구성한 12개 챕터 심층 요약", sub_title_style))
    elements.append(Spacer(1, 200))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=10))
    elements.append(Paragraph("작성자: Antigravity AI Project Engine", body_style))
    elements.append(Paragraph("날짜: 2026.01.23", body_style))
    elements.append(PageBreak())

    # --- Part 1 ---
    elements.append(Paragraph("Part 1: 머신러닝의 기초와 철학 (Ch. 1 - 2)", part_style))
    
    elements.append(Paragraph("1장. 머신러닝의 정의와 범주", chapter_style))
    elements.append(Paragraph(
        "통계적 학습(Statistical Learning)은 단순히 예측을 넘어 데이터를 깊게 이해하기 위한 도구들의 집합입니다. "
        "전통적인 통계학이 가설 검정에 집중했다면, 머신러닝은 데이터로부터 스스로 패턴을 찾아내는 '학습'에 초점을 맞춥니다. "
        "지도 학습은 정답이 있는 문제(회귀, 분류)를 풀고, 비지도 학습은 데이터 내부의 숨겨진 구조(군집화)를 탐색합니다.", 
        body_style))

    elements.append(Paragraph("2장. 모델 선택의 핵심 전략 (The Bias-Variance Tradeoff)", chapter_style))
    elements.append(Paragraph(
        "머신러닝의 성패는 '편향(Bias)'과 '분산(Variance)' 사이의 아슬아슬한 줄타기에 달려 있습니다. "
        "편향이 높으면 모델이 너무 단순해져(Underfitting) 중요 패턴을 놓치고, 분산이 높으면 훈련 데이터에 너무 집착하여(Overfitting) 새로운 데이터에서 실패합니다. "
        "시니어 분석가라면 훈련 오차가 0에 가깝다고 기뻐하는 대신, 테스트 오차가 최소화되는 지점을 찾아내는 '냉철함'을 유지해야 합니다.",
        body_style))

    # --- Part 2 ---
    elements.append(Paragraph("Part 2: 예측 모델링의 핵심 기법 (Ch. 3 - 4)", part_style))
    
    elements.append(Paragraph("3장. 선형 회귀 (Linear Regression)", chapter_style))
    elements.append(Paragraph(
        "가장 오래되었지만 가장 강력한 도구입니다. 변수 간의 선형적 인과관계를 수치로 증명합니다. "
        "단순 매출액 계산이 아니라, '광고비 1원 투입 시 매출이 몇 원 늘어나는가'를 설명할 수 있는 유일한 모형이기도 합니다. "
        "p-value와 결정계수(R-squared)를 통해 모델의 유의성을 철저히 검증해야 합니다.", 
        body_style))

    elements.append(Paragraph("4장. 분류 기법 (Classification)", chapter_style))
    elements.append(Paragraph(
        "현실 세계의 비즈니스 질문은 대개 '예/아니오'로 나뉩니다. 로지스틱 회귀는 이런 이분법적 사고를 확률로 변환해 줍니다. "
        "또한 판별 분석(LDA)은 클래스 간의 변동을 최소화하는 경계를 찾아냅니다. "
        "주의할 점은 클래스 불균형입니다. 희귀한 사건(예: 사기 거래)을 탐지할 때는 단순 정확도가 아닌 정밀도와 재현율을 봐야 합니다.", 
        body_style))

    # --- Part 3 ---
    elements.append(Paragraph("Part 3: 모델의 완성도 높이기 (Ch. 5 - 6)", part_style))
    
    elements.append(Paragraph("5장. 재표본 추출 (Resampling)", chapter_style))
    elements.append(Paragraph(
        "데이터를 믿지 말고 검증 시스템을 믿으십시오. 교차 검증(Cross-Validation)은 데이터를 쪼개어 모델의 실력을 다각도로 테스트합니다. "
        "특히 관측치가 적은 실무 데이터에서 부트스트랩을 활용해 통계량의 신뢰 구간을 구하는 것은 시니어의 기본 덕목입니다.", 
        body_style))

    elements.append(Paragraph("6장. 모델 선택 및 정규화 (Shrinkage)", chapter_style))
    elements.append(Paragraph(
        "때로는 모든 변수를 쓰는 것이 독이 됩니다. 릿지(Ridge)와 라쏘(Lasso) 기법은 불필요한 변수의 영향력을 줄입니다. "
        "특히 라쏘는 중요하지 않은 변수를 완전히 0으로 만들어버려, 데이터 속의 진짜 주인공만 남기는 '변수 선택' 기능을 수행합니다.", 
        body_style))

    # --- Part 4 ---
    elements.append(Paragraph("Part 4: 복잡한 데이터 대응 기술 (Ch. 7 - 9)", part_style))
    
    elements.append(Paragraph("7장. 비선형 모델링 (Beyond Linearity)", chapter_style))
    elements.append(Paragraph(
        "현실은 직선으로 설명되지 않는 곡선의 세계입니다. 다항 회귀와 스플라인 기법은 데이터의 굴곡을 따라가는 부드러운 모델을 만듭니다. "
        "다만 굴곡을 너무 세밀하게 따라가면 과적합의 함정에 빠질 수 있음을 항상 경계하십시오.", 
        body_style))

    elements.append(Paragraph("8장. 트리 기반 방법 (Tree-Based)", chapter_style))
    elements.append(Paragraph(
        "나무 하나는 약하지만, 숲은 강력합니다. 랜덤 포레스트는 여러 나무의 투표로 결론을 내어 안정적인 성능을 보장합니다. "
        "반면 부스팅 기법은 앞선 모델의 실수를 보완하며 독보적인 정확도를 추구합니다. 현대 정형 데이터 분석의 '끝판왕'이라 할 수 있습니다.", 
        body_style))

    elements.append(Paragraph("9장. 서포트 벡터 머신 (SVM)", chapter_style))
    elements.append(Paragraph(
        "데이터를 가로지르는 가장 넓은 길(Margin)을 찾아내는 기법입니다. "
        "비선형 데이터도 '커널 트릭'을 통해 고차원으로 변환하여 깔끔한 분류 경계를 만들어냅니다.", 
        body_style))

    # --- Part 5 ---
    elements.append(Paragraph("Part 5: 현대적 인공지능과 비지도 학습 (Ch. 10 - 12)", part_style))
    
    elements.append(Paragraph("10장. 딥러닝 (Deep Learning)", chapter_style))
    elements.append(Paragraph(
        "비정형 데이터(이미지, 영상, 텍스트)의 시대를 연 주인공입니다. "
        "특장점 추출을 수동으로 하지 않고 신경망이 스스로 학습합니다. "
        "성능은 압도적이지만 '왜 그런 결과가 나왔는지' 설명하기 어려운 블랙박스 특성을 이해하고 다뤄야 합니다.", 
        body_style))

    elements.append(Paragraph("11장. 생존 분석 (Survival Analysis)", chapter_style))
    elements.append(Paragraph(
        "이탈 예측에서 더 나아가 '언제 이탈할 것인가'를 다룹니다. "
        "부도 시점, 장비 고전 시점, 고객 생애 가치 분석 등에 필수적인 고급 기법입니다.", 
        body_style))

    elements.append(Paragraph("12장. 비지도 학습 (Unsupervised)", chapter_style))
    elements.append(Paragraph(
        "정답이 없는 데이터에서 보물을 찾는 기술입니다. PCA로 차원을 축소하여 시각화하거나, "
        "K-평균 군집화를 통해 비슷한 고객들을 묶어 마케팅 전략을 수립합니다. 모든 데이터 탐색의 시작점입니다.", 
        body_style))

    # --- Conclusion ---
    elements.append(Spacer(1, 50))
    elements.append(Paragraph(
        "시니어 분석가의 최종 코멘트: "
        "이 방대한 머신러닝의 지도를 여행할 때 길을 잃지 않는 법은 딱 하나입니다. "
        "기존의 지식에 매몰되지 않고, 항상 데이터의 목소리에 귀를 기울이며 '왜?'라고 묻는 것입니다. "
        "이 리포트가 사용자님의 '지휘자'로서의 첫걸음에 든든한 등대가 되길 바랍니다.", 
        quote_style))

    # PDF 빌드
    doc.build(elements)
    print(f"✅ 상세 머신러닝 요약 PDF 생성 완료: {output_path}")

if __name__ == "__main__":
    output_file = "/Users/sebokoh/Desktop/Detailed_ML_Summary_ISL_Final.pdf"
    create_detailed_ml_pdf(output_file)
