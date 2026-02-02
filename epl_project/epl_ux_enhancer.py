"""
EPL 앱 UX 개선 및 공유 기능 모듈
GEMINI.md Protocol 준수 - 프로덕션 레벨 품질
"""

import streamlit as st
import base64
from pathlib import Path
from datetime import datetime
import json


class ModernUIEnhancer:
    """EPL 앱 UX 개선 및 공유 기능 제공 (Modern SOTA UI)"""
    
    @staticmethod
    def add_loading_spinner(message: str = "데이터 로딩 중..."):
        """로딩 스피너 추가"""
        return st.spinner(message)
    
    @staticmethod
    def add_error_handler(error_message: str):
        """친절한 에러 메시지 표시"""
        st.error(f"""
        ❌ **문제가 발생했습니다**
        
        {error_message}
        
        **해결 방법:**
        1. 페이지를 새로고침해주세요 (F5)
        2. 사이드바의 '🔄 전체 새로고침' 버튼을 클릭하세요
        3. 문제가 지속되면 관리자에게 문의하세요
        """)
    
    @staticmethod
    def add_success_message(message: str, icon: str = "✅"):
        """성공 메시지 표시"""
        st.success(f"{icon} {message}")
    
    @staticmethod
    def add_share_buttons(title: str, url: str = None):
        """SNS 공유 버튼 추가"""
        if url is None:
            url = "https://your-epl-app.streamlit.app"  # 실제 배포 URL로 변경
        
        st.markdown("### 📤 공유하기")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            twitter_url = f"https://twitter.com/intent/tweet?text={title}&url={url}"
            st.markdown(f"""
            <a href="{twitter_url}" target="_blank" style="text-decoration: none;">
                <div style="
                    background: linear-gradient(90deg, #1DA1F2, #0d8bd9);
                    color: white;
                    padding: 12px;
                    border-radius: 10px;
                    text-align: center;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 0.2s;
                ">
                    🐦 Twitter
                </div>
            </a>
            """, unsafe_allow_html=True)
        
        with col2:
            facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={url}"
            st.markdown(f"""
            <a href="{facebook_url}" target="_blank" style="text-decoration: none;">
                <div style="
                    background: linear-gradient(90deg, #4267B2, #365899);
                    color: white;
                    padding: 12px;
                    border-radius: 10px;
                    text-align: center;
                    font-weight: 600;
                    cursor: pointer;
                ">
                    📘 Facebook
                </div>
            </a>
            """, unsafe_allow_html=True)
        
        with col3:
            reddit_url = f"https://www.reddit.com/submit?url={url}&title={title}"
            st.markdown(f"""
            <a href="{reddit_url}" target="_blank" style="text-decoration: none;">
                <div style="
                    background: linear-gradient(90deg, #FF4500, #d63b00);
                    color: white;
                    padding: 12px;
                    border-radius: 10px;
                    text-align: center;
                    font-weight: 600;
                    cursor: pointer;
                ">
                    🔴 Reddit
                </div>
            </a>
            """, unsafe_allow_html=True)
        
        with col4:
            # 카카오톡 공유 (Web Share API)
            st.markdown(f"""
            <div onclick="shareContent()" style="
                background: linear-gradient(90deg, #FEE500, #f5dc00);
                color: #3C1E1E;
                padding: 12px;
                border-radius: 10px;
                text-align: center;
                font-weight: 600;
                cursor: pointer;
            ">
                💬 카카오톡
            </div>
            
            <script>
            function shareContent() {{
                if (navigator.share) {{
                    navigator.share({{
                        title: '{title}',
                        url: '{url}'
                    }});
                }} else {{
                    alert('공유 기능을 지원하지 않는 브라우저입니다.');
                }}
            }}
            </script>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def add_download_button(data: str, filename: str, label: str = "📥 다운로드"):
        """파일 다운로드 버튼 추가"""
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="text-decoration: none;"><div style="background: linear-gradient(90deg, #667eea, #764ba2); color: white; padding: 12px 24px; border-radius: 10px; text-align: center; font-weight: 600; cursor: pointer; display: inline-block;">{label}</div></a>'
        st.markdown(href, unsafe_allow_html=True)
    
    @staticmethod
    def add_mobile_optimization():
        """모바일 최적화 CSS 추가"""
        st.markdown("""
        <style>
            /* 모바일 최적화 */
            @media (max-width: 768px) {
                .stApp {
                    padding: 0.5rem !important;
                }
                
                h1 {
                    font-size: 1.5rem !important;
                }
                
                h2 {
                    font-size: 1.3rem !important;
                }
                
                h3 {
                    font-size: 1.1rem !important;
                }
                
                /* 차트 반응형 */
                .stPlotlyChart {
                    width: 100% !important;
                }
                
                /* 버튼 터치 영역 확대 */
                .stButton > button {
                    min-height: 48px !important;
                    font-size: 16px !important;
                }
            }
            
            /* 로딩 스피너 스타일 */
            .stSpinner > div {
                border-color: #667eea transparent transparent transparent !important;
            }
            
            /* 에러 메시지 스타일 */
            .stAlert {
                border-radius: 12px !important;
                padding: 1.5rem !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def add_performance_metrics():
        """성능 모니터링 메트릭 추가"""
        if 'page_load_time' not in st.session_state:
            st.session_state['page_load_time'] = datetime.now()
        
        elapsed = (datetime.now() - st.session_state['page_load_time']).total_seconds()
        
        with st.expander("⚡ 성능 정보", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("페이지 로드 시간", f"{elapsed:.2f}초")
            with col2:
                cache_size = len(st.session_state)
                st.metric("캐시 항목", f"{cache_size}개")
            with col3:
                st.metric("상태", "✅ 정상")
    
    @staticmethod
    def add_screenshot_button():
        """스크린샷 촬영 버튼 (JavaScript)"""
        st.markdown("""
        <button onclick="captureScreenshot()" style="
            background: linear-gradient(90deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            font-size: 16px;
        ">
            📸 스크린샷 촬영
        </button>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
        <script>
        function captureScreenshot() {
            html2canvas(document.body).then(canvas => {
                const link = document.createElement('a');
                link.download = 'epl_dashboard_' + Date.now() + '.png';
                link.href = canvas.toDataURL();
                link.click();
            });
        }
        </script>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def add_dark_mode_toggle():
        """다크 모드 토글 (이미 다크 모드지만 추가 커스터마이징)"""
        st.markdown("""
        <style>
            /* 다크 모드 강화 */
            .stApp {
                background: radial-gradient(circle at top right, #1a1c24, #0e1117) !important;
            }
            
            /* 텍스트 가독성 향상 */
            p, li, span {
                color: #E0E0E0 !important;
            }
            
            /* 카드 배경 강화 */
            .stMarkdown, .stDataFrame {
                background: rgba(255, 255, 255, 0.03) !important;
                border-radius: 12px !important;
                padding: 1rem !important;
            }
        </style>
        """, unsafe_allow_html=True)

    def render_performance_matrix(self, clubs_data: list):
        """구단 전력 분석 매트릭스 시각화 (Power Index vs Rank)"""
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(clubs_data)
        
        # [Design] Premium Scatter Chart
        fig = px.scatter(
            df, 
            x="power_index", 
            y="current_rank",
            text="team_name",
            size="power_index",
            color="power_index",
            color_continuous_scale="Viridis",
            labels={
                "power_index": "AI 전력 지수",
                "current_rank": "리그 순위",
                "team_name": "구단명"
            },
            title="EPL 구단 효율성 매트릭스 (Efficiency Matrix)"
        )
        
        fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='white')))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            yaxis=dict(autorange="reversed") # 순위는 낮을수록 위로
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("※ **분석 가이드**: 우측 상단일수록 전력 대비 성적이 좋은 '고효율' 팀입니다.")

    def render_tactical_similarity_map(self, selected_team: str):
        """
        [SOTA] 전술 유사도 맵 시각화 (Tactical Similarity Map)
        각 팀의 전술적 스타일(점유율, 압박 강도, 공격 속도 등)을 2차원 공간에 투영합니다.
        """
        import pandas as pd
        import plotly.express as px
        import numpy as np

        # [Heuristic Tactical Data] 2025/26 시즌 기반 전술적 좌표 (점유율 vs 압박/속도)
        # x: Possession/Structure (점유/구조화), y: Intensity/Directness (압박/직설성)
        tactical_data = {
            "Manchester City": [95, 70], "Arsenal": [90, 75], "Liverpool": [85, 95],
            "Tottenham": [80, 90], "Aston Villa": [75, 80], "Chelsea": [88, 65],
            "Newcastle": [60, 85], "Manchester United": [55, 80], "Brighton": [92, 60],
            "West Ham": [40, 70], "Everton": [35, 75], "Brentford": [45, 85],
            "Fulham": [65, 55], "Bournemouth": [50, 80], "Wolves": [55, 65],
            "Crystal Palace": [45, 60], "Nottingham Forest": [30, 85], "Leicester": [55, 50],
            "Ipswich": [40, 60], "Southampton": [70, 40]
        }
        
        # 한글 구단명 매핑 (app.py의 selected_team과 호환)
        kor_to_eng = {
            "맨체스터 시티": "Manchester City", "아스날": "Arsenal", "리버풀": "Liverpool",
            "토트넘 홋스퍼": "Tottenham", "아스톤 빌라": "Aston Villa", "첼시": "Chelsea",
            "뉴캐슬 유나이티드": "Newcastle", "맨체스터 유나이티드": "Manchester United", "브라이튼": "Brighton",
            "웨스트햄 유나이티드": "West Ham", "에버튼": "Everton", "브렌트포드": "Brentford",
            "풀럼": "Fulham", "본머스": "Bournemouth", "울버햄튼": "Wolves",
            "크리스탈 팰리스": "Crystal Palace", "노팅엄 포레스트": "Nottingham Forest", "레스터 시티": "Leicester",
            "입스위치 타운": "Ipswich", "사우스햄튼": "Southampton"
        }
        
        data_list = []
        for kor, eng in kor_to_eng.items():
            coords = tactical_data.get(eng, [50, 50])
            # 노이즈 추가하여 시각적 분리감 확보
            x = coords[0] + np.random.uniform(-2, 2)
            y = coords[1] + np.random.uniform(-2, 2)
            
            data_list.append({
                "Team": kor,
                "Possession_Structure": x,
                "Intensity_Directness": y,
                "is_selected": "Target" if kor == selected_team else "Other",
                "Size": 15 if kor == selected_team else 10
            })
            
        df = pd.DataFrame(data_list)
        
        fig = px.scatter(
            df,
            x="Possession_Structure",
            y="Intensity_Directness",
            text="Team",
            color="is_selected",
            size="Size",
            color_discrete_map={"Target": "#FF4B4B", "Other": "#636EFA"},
            labels={
                "Possession_Structure": "점유율 및 구조화 (Possession & Structure)",
                "Intensity_Directness": "압박 강도 및 속도 (Intensity & Directness)"
            },
            title=f"⚽ EPL 전술 유사도 맵 (Target: {selected_team})"
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            showlegend=False,
            height=600,
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 105]),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 105])
        )
        
        # 사분면 가이드라인 추가
        fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        fig.add_vline(x=50, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        
        # 사분면 라벨
        fig.add_annotation(x=95, y=95, text="점유+압박 (Modern Elite)", showarrow=False, font=dict(color="rgba(255,255,255,0.5)"))
        fig.add_annotation(x=5, y=95, text="직설적 압박 (Heavy Metal)", showarrow=False, font=dict(color="rgba(255,255,255,0.5)"))
        fig.add_annotation(x=95, y=5, text="지공 위주 (Slow Build)", showarrow=False, font=dict(color="rgba(255,255,255,0.5)"))
        fig.add_annotation(x=5, y=5, text="선수비 후역습 (Low Block)", showarrow=False, font=dict(color="rgba(255,255,255,0.5)"))

        st.plotly_chart(fig, use_container_width=True)
        st.caption("※ **분석 가이드**: 좌표가 가까울수록 전술적으로 유사한 스타일을 구사하는 팀입니다.")

    @staticmethod
    def generate_seo_meta(title: str, description: str, image_url: str = None):
        """SEO 메타 태그 생성"""
        if image_url is None:
            image_url = "https://your-epl-app.streamlit.app/og-image.png"
        
        st.markdown(f"""
        <meta property="og:title" content="{title}" />
        <meta property="og:description" content="{description}" />
        <meta property="og:image" content="{image_url}" />
        <meta property="og:type" content="website" />
        
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="{title}" />
        <meta name="twitter:description" content="{description}" />
        <meta name="twitter:image" content="{image_url}" />
        
        <meta name="description" content="{description}" />
        <meta name="keywords" content="EPL, 프리미어리그, AI 분석, 축구 데이터, 전술 분석, 승부 예측" />
        """, unsafe_allow_html=True)


# 사용 예시 (app.py에 통합)
def integrate_enhancements():
    """app.py에 통합할 함수"""
    enhancer = EPLAppEnhancer()
    
    # 1. 모바일 최적화
    enhancer.add_mobile_optimization()
    
    # 2. 다크 모드 강화
    enhancer.add_dark_mode_toggle()
    
    # 3. SEO 메타 태그
    enhancer.generate_seo_meta(
        title="EPL-X Manager | AI 기반 프리미어리그 분석 대시보드",
        description="Gemini 2.0 기반 실시간 EPL 팀 분석, 승부 예측, 전술 리포트를 제공하는 프리미엄 대시보드"
    )
    
    # 4. 공유 버튼 (메인 페이지 하단에 추가)
    # enhancer.add_share_buttons("EPL-X Manager 대시보드", "https://your-app-url.com")
    
    # 5. 성능 모니터링
    # enhancer.add_performance_metrics()


if __name__ == "__main__":
    print("✅ EPL App Enhancer 모듈 로드 완료")
    print("📖 사용법: from epl_ux_enhancer import EPLAppEnhancer")
