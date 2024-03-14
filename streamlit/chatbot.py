import requests
import json

def chatbot(data:dict):
    url = 'http://223.130.137.53:8080/generate/'  # FastAPI 서버의 엔드포인트 URL
    response = requests.post(url, data=json.dumps(data))  # GET 요청을 보냄

    if response.status_code == 200:  # 요청이 성공했을 때
        data = response.json()  # JSON 형식의 응답 데이터를 파이썬 객체로 변환
        answer = data['answer']
        return answer
    else:
        print("Failed to fetch data:", response.status_code)
        return None