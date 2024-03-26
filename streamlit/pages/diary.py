from datetime import datetime
from collections import Counter
import streamlit as st
import json
import sqlite3
import pandas as pd
from PIL import Image

from menu import menu_with_redirect
from chatbot import *
from login import load_user_data

def main():
  menu_with_redirect()

  if 'count' not in st.session_state:
    st.session_state.count = 1
  if 'messages' not in st.session_state:
    st.session_state.messages = [{'generation_id': st.session_state.count, 'role': 'assistant', 'content': '오늘 하루는 어땠어?'}]

  st.title(f'{datetime.now().year}년 {datetime.now().month}월 {datetime.now().day}일 오늘의 일기')
  st.divider()


  if st.session_state['my_data'][st.session_state['my_data']['date']==str(st.session_state['today'])].empty:
    prompt = st.chat_input()
  else:
    prompt = st.chat_input(disabled=True)
    st.session_state.messages = json.loads(st.session_state['my_data'][st.session_state['my_data']['date']==str(st.session_state['today'])].content.values[0])
    st.session_state['count'] = st.session_state.messages[-1]['generation_id']
    
  for message in st.session_state.messages:
    if message['role'] == 'user':
      with st.chat_message(message['role'], avatar="🧑"):
          st.markdown(message["content"])
    else:
      with st.chat_message(message['role'], avatar="🐥"):
          st.markdown(message["content"])
   
  if prompt:
    st.chat_message("user", avatar="🧑").markdown(prompt)    
    data = {
      "generation_id" : st.session_state.count, # generation task id
      "query" : prompt, # 주어진 질문
      "history" : st.session_state.messages
    }
    st.session_state.count += 1

    with st.chat_message("assistant", avatar="🐥"):
      with st.spinner('답변 생성중'):
        response = call_api(st.secrets['chatbot_url'],data)
        
      st.session_state.messages.append({'generation_id': st.session_state.count, 'role': 'user', 'content': prompt})
      st.session_state.count += 1
      st.session_state.messages.append({'generation_id': st.session_state.count, "role": "assistant", "content": response})
      st.markdown(response)

  if st.session_state.count >= 9 and st.session_state['my_data'][st.session_state['my_data']['date']==str(st.session_state['today'])].empty:
    with st.chat_message("assistant", avatar="✍️"):
      st.markdown('오늘의 일기를 정리해줄까? 아니면 더 이야기해도 좋아!')
      summary_btn = st.button('일기 생성 🧙')
    if summary_btn:
      with st.chat_message("assistant", avatar="🐥"):
        with st.spinner('Dr.부덕이가 일기를 생성중...'):
          st.session_state.count += 1
          data = {
            "generation_id": st.session_state.count,
            "query" : st.session_state['messages'],
          }
          summary = call_api(st.secrets['summary_url'], data)
          data = {
            "summary": summary
          }
          emotion = call_emotion_api(st.secrets['emotion_url'], data)
          emotion = str(emotion).replace("'", '"')
          data = {
            "chat": st.session_state.messages
          }
          word = call_word_api(st.secrets['word_url'], data)
          word = str(word).replace("'", '"')
      c.execute('INSERT INTO diarytable(diary_id, id, date, content, summary, emotion, word) VALUES (?,?,?,?,?,?,?)',(f"{datetime.today().strftime('%y%m%d')}_{st.session_state['id']}",
                                                                                                                      st.session_state['id'],
                                                                                                                      st.session_state['today'],
                                                                                                                      str(st.session_state['messages']).replace("'generation_id'", '"generation_id"').replace("'role'", '"role"').replace("'user'", '"user"').replace("'assistant'", '"assistant"').replace("'content': '", '"content": "').replace("'}", '"}'),
                                                                                                                      summary,
                                                                                                                      emotion,
                                                                                                                      word))
      diary.commit()
      st.session_state['my_data'] = load_user_data(c, st.session_state['id'])
      st.switch_page('pages/summary.py')
    
          
if __name__ == '__main__':
  diary = sqlite3.connect('diary.db')
  c = diary.cursor()

  main()