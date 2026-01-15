
            # [VISUALIZATION] SHAP 스타일 변수 중요도 시각화 (Mockup)
            st.markdown("### 📊 AI 변수 중요도 (SHAP Analysis)")
            st.markdown("어떤 요인이 이 승부의 향방을 결정했는지 AI가 인과관계를 분석했습니다.")
            
            # 가상 SHAP 값 생성 (시나리오별)
            import pandas as pd
            import altair as alt
            
            shap_data = pd.DataFrame({
                'Feature': ['홈 어드밴티지', '최근 득점력', '상대 전적', '부상자 영향', '감독 전술'],
                'Impact': [prob - 50, (h_data.get('goals_scored', 0) - 20)/2, 5.0 if h_power > a_power else -5.0, -3.0, 2.0],
                'Color': ['#4CAF50' if x > 0 else '#E91E63' for x in [prob - 50, (h_data.get('goals_scored', 0) - 20)/2, 5.0 if h_power > a_power else -5.0, -3.0, 2.0]]
            })
            
            chart = alt.Chart(shap_data).mark_bar().encode(
                x=alt.X('Impact', title='승리 기여도 (Impact)'),
                y=alt.Y('Feature', sort='-x', title='분석 변수'),
                color=alt.Color('Color', scale=None),
                tooltip=['Feature', 'Impact']
            ).properties(
                height=300
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            st.caption("※ 빨간색(Neg)은 패배/실점 요인, 초록색(Pos)은 승리/득점 요인을 의미합니다.")
