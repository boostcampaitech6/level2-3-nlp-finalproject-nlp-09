from datetime import datetime
from collections import Counter
import streamlit as st
import json
import sqlite3
import pandas as pd

from menu import menu_with_redirect
from chatbot import *

def main():
  menu_with_redirect()
  today = datetime.now().date()

  if 'count' not in st.session_state:
    st.session_state.count = 1
  if 'messages' not in st.session_state:
    st.session_state.messages = [{'generation_id': st.session_state.count, 'role': 'assistant', 'content': '오늘 하루는 어땠어?'}]

  st.title(f'{datetime.now().year}년 {datetime.now().month}월 {datetime.now().day}일 오늘의 일기')
  st.divider()


  if st.session_state.today_data.empty:
    prompt = st.chat_input()
  else:
    prompt = st.chat_input(disabled=True)
    st.session_state.messages = json.loads(st.session_state.today_data.content.values[0])
    st.session_state['count'] = st.session_state.messages[-1]['generation_id']
    
  for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])     
   
  if prompt:
    data = {
      "generation_id" : st.session_state.count, # generation task id
      "query" : prompt, # 주어진 질문
      "history" : st.session_state.messages
    }
    st.session_state.count += 1
    st.chat_message("user").markdown(prompt)    

    with st.chat_message("assistant"):
      with st.spinner('답변 생성중'):
        response = call_api(st.secrets['chatbot_url'],data)
      st.session_state.messages.append({'generation_id': st.session_state.count, 'role': 'user', 'content': prompt})
      st.session_state.count += 1
      st.session_state.messages.append({'generation_id': st.session_state.count, "role": "assistant", "content": response})
      st.markdown(response)

  if st.session_state.count == 5 and st.session_state.today_data.empty:
    with st.chat_message("assistant"):
      st.markdown('내용을 요약하시겠습니까?')
      summary_btn = st.button('요약하기')
    if summary_btn:
      with st.chat_message("assistant"):
        with st.spinner('요약중'):
          st.session_state.count += 1
          data = {
            "generation_id": st.session_state.count, # generation task id
            "query" : st.session_state['messages'], # 주어진 질문
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
          word = str(word ).replace("'", '"')
      c.execute('INSERT INTO diarytable(diary_id, id, date, content, summary, emotion, word) VALUES (?,?,?,?,?,?,?)',(f"{datetime.today().strftime('%y%m%d')}_{st.session_state['id']}",
                                                                                                                      st.session_state['id'],
                                                                                                                      today,
                                                                                                                      str(st.session_state['messages']).replace("'generation_id'", '"generateion_id"').replace("'role'", '"role"').replace("'user'", '"user"').replace("'assistant'", '"assistant"').replace("'content': '", '"content": "').replace("'}", '"}'),
                                                                                                                      summary,
                                                                                                                      emotion,
                                                                                                                      word))
      diary.commit()
      st.session_state['today_data'] = pd.DataFrame(data=[[f"{datetime.today().strftime('%y%m%d')}_{st.session_state['id']}",
                                                          st.session_state['id'],
                                                          today,
                                                          str(st.session_state['messages']).replace("'generation_id'", '"generateion_id"').replace("'role'", '"role"').replace("'user'", '"user"').replace("'assistant'", '"assistant"').replace("'content': '", '"content": "').replace("'}", '"}'),
                                                          summary,
                                                          emotion,
                                                          word]],
                                                    columns=st.session_state['my_data'].columns.to_list())
      st.switch_page('pages/summary.py')
    
          
if __name__ == '__main__':
  diary = sqlite3.connect('diary.db')
  c = diary.cursor()

  main()