# [FOUNDER STRATEGY] '기능'이 아닌 '통찰'을 공유하게 유도 (Lenny's Insight)
st.divider()
st.subheader("📤 분석 결과(Insight) 공유하기")

try:
    # 단순 홍보가 아닌 '데이터 증거' 위주로 텍스트 구성
    share_pred_text = f"""[AI 분석 리포트] {home} vs {away}
    
"오늘 경기는 {prob:.1f}% 확률로 {home if prob > 50 else away}의 우세가 점쳐집니다."

🛡️ 분석 요약:
{consensus.split(':')[1].strip() if ':' in consensus else consensus}

🚩 리스크 탐지:
{risk_msgs[0] if risk_msgs else "특이 리스크 없음 (Clean Condition)"}

📊 데이터 근거 상세 보기:
https://epl-data-2026.streamlit.app/

#EPL #AI분석 #축구데이터 #{home.replace(' ', '')} #{away.replace(' ', '')}"""
except:
     share_pred_text = f"분석을 먼저 실행해주세요."

st.info("💡 커뮤니티(레딧, 단톡방) 공유 팁: 앱 홍보보다 '데이터 분석 결과' 그 자체를 논쟁의 근거로 사용하면 호응이 훨씬 좋습니다.")
st.code(share_pred_text, language="text")

# Native Web Share
js_pred_text = share_pred_text.replace('\n', '\\n').replace("'", "\\'")

share_match_html = f"""
<style>
    .share-btn-match {{
        background: linear-gradient(135deg, #FEE500 0%, #F5D100 100%);
        color: #191919;
        border: none;
        padding: 14px 24px;
        text-align: center;
        display: block;
        font-size: 17px;
        font-weight: 800;
        margin: 10px 0;
        cursor: pointer;
        border-radius: 16px;
        width: 100%;
        box-shadow: 0 4px 15px rgba(254, 229, 0, 0.3);
        transition: all 0.2s;
    }}
    .share-btn-match:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(254, 229, 0, 0.4); }}
</style>

<button class="share-btn-match" onclick="nativeShareMatch()">
    🟡 카카오톡/SNS로 통찰 공유하기
</button>

<script>
function nativeShareMatch() {{
    if (navigator.share) {{
        navigator.share({{
            title: '{home} vs {away} AI 리포트',
            text: '{js_pred_text}',
            url: 'https://epl-data-2026.streamlit.app/'
        }})
        .then(() => console.log('Successful share'))
        .catch((error) => console.log('Error sharing', error));
    }} else {{
        alert('⚠️ 모바일 환경에서 [공유] 버튼을 사용하거나, 위 텍스트를 복사해서 사용하세요!');
    }}
}}
</script>
"""
st.markdown(share_match_html, unsafe_allow_html=True)
