from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# 1. 한글 폰트 등록 (Mac 전용 AppleGothic)
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
pdfmetrics.registerFont(TTFont('AppleGothic', FONT_PATH))

def create_pdf(output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # 2. 커스텀 스타일 정의
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='AppleGothic',
        fontSize=20,
        spaceAfter=20,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontName='AppleGothic',
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        color=colors.navy
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='AppleGothic',
        fontSize=10,
        leading=15,
        spaceAfter=8
    )

    elements = []

    # 3. 내용 구성
    elements.append(Paragraph("자동차 등록 데이터 분석 전략 제언 (Phase 1)", title_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("본 계획은 데이터의 숨겨진 가치를 발굴하기 위한 가설과 시각화 계획을 담고 있습니다.", body_style))
    
    elements.append(Paragraph("1. 분석의 핵심 목표", heading_style))
    elements.append(Paragraph("단순한 차량 등록 현황 파악을 넘어, <b>[시장 변화의 시그널]</b>을 포착하고 <b>[고객 행동의 인과관계]</b>를 규명하여 비즈니스 의사결정에 기여합니다.", body_style))
    
    elements.append(Paragraph("2. EDA 가설 및 분석 테마", heading_style))
    
    themes = [
        ("Theme A: 시장 전이 및 브랜드 포지셔닝", "수입 프리미엄 브랜드의 성장과 법인 고객 중심 가속화 분석. 전기차 전환 시점의 차급 상향 평준화 현상 규명."),
        ("Theme B: 세그먼트 디스크립터", "마스킹된 시군구/인구 데이터 기반 핵심 구매층 거주지 및 선호 브랜드 상관관계 분석."),
        ("Theme C: 에너지 패러다임 쉬프트", "전기차 출력 데이터 패턴을 통한 고성능 럭셔리 시장 이동성 및 보조금 집행 시기 연동성 분석.")
    ]
    
    for title, desc in themes:
        elements.append(Paragraph(f"• <b>{title}</b>", body_style))
        elements.append(Paragraph(f"  {desc}", body_style))
        elements.append(Spacer(1, 5))

    elements.append(Paragraph("3. 시각화 및 리포트 구성 (Visualization Index)", heading_style))
    
    table_data = [
        ["섹션", "핵심 지표 (KPI)", "기대 효과"],
        ["Executive Summary", "전체 등록 수, 전월 대비 증감률", "프로젝트 규모 및 성장성 파악"],
        ["Brand War-room", "제조사별 시장 점유율 (MS)", "브랜드 간 경쟁 구도 변화 체감"],
        ["Deep Demo-Analysis", "연령/성별/지역별 등록 분포", "핵심 소비자 페르소나 확인"],
        ["Future Signal", "전기차 vs 내연기관 성장 속도", "시장의 미래 방향성 제시"]
    ]
    
    table = Table(table_data, colWidths=[100, 200, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'AppleGothic'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    elements.append(Paragraph("4. 향후 로드맵 제언", heading_style))
    roadmap = [
        "Phase 1 (현재): 가설 기반 데이터 뼈대 구축 및 기초 EDA 결과 공유",
        "Phase 2 (심화): 의뢰인 피드백 반영 및 고급 통계 검정을 통한 가설 증명",
        "Phase 3 (예측): 딥러닝(TimesFM 등) 기반 다음 분구 등록량 예측 및 전략 리포트 완성"
    ]
    for item in roadmap:
        elements.append(Paragraph(f"- {item}", body_style))

    # PDF 생성
    doc.build(elements)
    print(f"✅ PDF 생성 완료: {output_path}")

if __name__ == "__main__":
    output_file = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/Kmong_Proposal_Phase1.pdf"
    create_pdf(output_file)
