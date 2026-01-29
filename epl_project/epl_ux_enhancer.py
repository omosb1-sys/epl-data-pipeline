"""
EPL 앱 UX 개선 및 공유 기능 모듈
GEMINI.md Protocol 준수 - 프로덕션 레벨 품질
"""

import streamlit as st
import base64
from pathlib import Path
from datetime import datetime
import json


class EPLAppEnhancer:
    """EPL 앱 UX 개선 및 공유 기능 제공"""
    
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
