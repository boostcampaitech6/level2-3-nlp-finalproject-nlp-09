import requests
import json

import streamlit as st

def call_api(url, data:dict):
    response = requests.post(url, data=json.dumps(data))  # GET 요청을 보냄

    if response.status_code == 200:  # 요청이 성공했을 때
        data = response.json()  # JSON 형식의 응답 데이터를 파이썬 객체로 변환
        answer = data['answer']
        return answer
    
def call_emotion_api(url, data:dict):
    response = requests.post(url, data=json.dumps(data))  # GET 요청을 보냄

    if response.status_code == 200:  # 요청이 성공했을 때
        data = response.json()  # JSON 형식의 응답 데이터를 파이썬 객체로 변환
        answer = data['emotion']
        return answer
    
def call_word_api(url, data:dict):
    response = requests.post(url, data=json.dumps(data))  # GET 요청을 보냄

    if response.status_code == 200:  # 요청이 성공했을 때
        data = response.json()  # JSON 형식의 응답 데이터를 파이썬 객체로 변환
        answer = data['word']
        return answer