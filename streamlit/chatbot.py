import requests
import json

import streamlit as st

def call_api(data:dict):
    url = st.secrets['api_url']  # FastAPI 서버의 엔드포인트 URL
    response = requests.post(url, data=json.dumps(data))  # GET 요청을 보냄

    if response.status_code == 200:  # 요청이 성공했을 때
        data = response.json()  # JSON 형식의 응답 데이터를 파이썬 객체로 변환
        answer = data['answer']
        return answer