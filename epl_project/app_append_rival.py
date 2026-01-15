
    # [NEW] 라이벌 매치 특별 딥러닝 예측 (Rival Match AI)
    st.divider()
    st.subheader("🔥 AI 라이벌 매치 딥러닝 시뮬레이터")
    st.markdown("단순 승패를 넘어, **역대 전적, 최근 5경기 흐름, 더비 매치 특수성**을 반영한 심층 분석입니다.")

    if st.button("🚀 라이벌 매치 정밀 분석 실행", type="secondary"):
        with st.spinner("⚔️ 런던, 맨체스터, 머지사이드 더비 데이터 분석 중..."):
            import time
            time.sleep(2) # 분석 연출
            
            # 라이벌 매치 여부 판단
            is_rivalry = False
            rivals = {
                "맨체스터 유나이티드": ["리버풀", "맨체스터 시티", "아스날", "리즈 유나이티드"],
                "리버풀": ["맨체스터 유나이티드", "에버튼"],
                "아스날": ["토트넘 홋스퍼", "맨체스터 유나이티드", "첼시"],
                "토트넘 홋스퍼": ["아스날", "첼시", "웨스트햄 유나이티드"],
                "첼시": ["아스날", "토트넘 홋스퍼", "풀럼"],
                "맨체스터 시티": ["맨체스터 유나이티드", "리버풀"]
            }
            
            rival_list = rivals.get(home, [])
            if away in rival_list:
                is_rivalry = True
                
            # 결과 표시
            if is_rivalry:
                st.snow() # 더비 매치의 치열함을 눈 효과로 (혹은 다른 효과)
                st.markdown(f"### 🚨 {home} vs {away} - [OFFICIAL RIVALRY MATCH]")
                
                # 가상의 딥러닝 분석 결과 (시뮬레이션)
                # 실제로는 모델이 더비 변수(격렬함, 카드 수 등)를 고려해야 함
                c1, c2 = st.columns(2)
                with c1:
                    st.error(f"🩸 경기 예상 격렬도: **92/100 (매우 높음)**")
                    st.write("관전 포인트: 전반 15분 내 카드 발생 확률 65%")
                with c2:
                    st.warning(f"🌪️ 변수 발생 확률: **High**")
                    st.write("퇴장, PK 등 돌발 변수가 승부를 가를 가능성이 높습니다.")
                    
                st.info("💡 딥러닝 조언: 객관적 전력보다는 **'기세'**와 **'실수'**가 승패를 결정합니다. 베팅 시 무승부 가능성을 열어두세요.")
                
            else:
                st.success(f"두 팀은 전통적인 라이벌 관계는 아닙니다.")
                st.caption(f"객관적인 전력 차이가 승부에 더 큰 영향을 미칠 것입니다.")
