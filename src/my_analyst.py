from openai import OpenAI

def 물어봐(질문):
    client = OpenAI(
        api_key="up_ehsQVOrLMLFGQN6jxe26k56jK09px",
        base_url="https://api.upstage.ai/v1/solar"
    )
    
    response = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=[
            {"role": "system", "content": "너는 전설적인 데이터 분석가야. 주니어에게 친절하고 전문적으로 설명해줘."},
            {"role": "user", "content": 질문}
        ]
    )
    return print(response.choices[0].message.content)