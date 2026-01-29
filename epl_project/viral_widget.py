import streamlit as st

def render_viral_card(home_team, away_team, home_logo, away_logo, win_prob, key_insight):
    """
    [Lenny's Widget-First] 
    SNS(카톡, 레딧, X) 공유에 최적화된 '프리미엄 전술 카드' 위젯.
    Glassmorphism과 고대비 색상을 사용하여 사용자의 공유 욕구를 자극합니다.
    """
    
    # 확률에 따른 색상 결정
    bar_color = "#FF4B4B" if win_prob > 50 else "#1E88E5"
    
    card_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&display=swap');
        
        .viral-card-container {{
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            border-radius: 24px;
            padding: 30px;
            color: white;
            font-family: 'Outfit', sans-serif;
            border: 2px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            max-width: 500px;
            margin: 20px auto;
            position: relative;
            overflow: hidden;
            text-align: center;
        }}
        
        .viral-card-container::before {{
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255, 75, 75, 0.1) 0%, transparent 70%);
            z-index: 0;
        }}

        .card-header {{
            font-weight: 900;
            font-size: 14px;
            letter-spacing: 3px;
            color: #94a3b8;
            margin-bottom: 20px;
            position: relative;
            z-index: 1;
        }}

        .teams-display {{
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin-bottom: 25px;
            position: relative;
            z-index: 1;
        }}

        .team-box {{
            flex: 1;
        }}

        .team-logo {{
            width: 70px;
            height: 70px;
            object-fit: contain;
            filter: drop-shadow(0 0 10px rgba(255,255,255,0.2));
            margin-bottom: 10px;
        }}

        .team-name {{
            font-weight: 700;
            font-size: 18px;
        }}

        .vs-badge {{
            background: rgba(255, 255, 255, 0.1);
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 900;
            font-size: 12px;
            color: #fca311;
        }}

        .result-section {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            z-index: 1;
        }}

        .win-title {{
            font-size: 16px;
            color: #cbd5e1;
            margin-bottom: 10px;
        }}

        .win-prob {{
            font-size: 48px;
            font-weight: 900;
            margin-bottom: 5px;
            background: linear-gradient(90deg, #fff, {bar_color});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .insight-tag {{
            display: inline-block;
            background: {bar_color};
            color: white;
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 700;
            margin-top: 10px;
            text-transform: uppercase;
        }}

        .footer-brand {{
            font-size: 10px;
            color: #64748b;
            margin-top: 20px;
            letter-spacing: 2px;
        }}
    </style>
    
    <div class="viral-card-container">
        <div class="card-header">AI MATCH ANALYSIS</div>
        <div class="teams-display">
            <div class="team-box">
                <img src="{home_logo}" class="team-logo">
                <div class="team-name">{home_team}</div>
            </div>
            <div class="vs-badge">VS</div>
            <div class="team-box">
                <img src="{away_logo}" class="team-logo">
                <div class="team-name">{away_team}</div>
            </div>
        </div>
        
        <div class="result-section">
            <div class="win-title">AI PREDICTED WIN PROBABILITY</div>
            <div class="win-prob">{win_prob:.1f}%</div>
            <div class="insight-tag">🎯 {key_insight}</div>
        </div>
        
        <div class="footer-brand">POWERED BY ANTIGRAVITY | EPL-X V12.0</div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # 공유 유도 텍스트
    st.caption("📱 위 카드를 캡처해서 단톡방이나 SNS에 공유해보세요!")
