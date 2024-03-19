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

  if "count" not in st.session_state:
    st.session_state.count = 1

  if "messages" not in st.session_state:
    st.session_state.messages = [{'generation_id': st.session_state.count, 'role': 'assistant', 'content': '오늘 하루는 어떠셨나요?'}]

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
    st.session_state.count += 1
    st.session_state.messages.append({'generation_id': st.session_state.count, 'role': 'user', 'content': prompt})
    st.chat_message("user").markdown(prompt)

    data = {
      "generation_id" : st.session_state.count, # generation task id
      "query" : prompt, # 주어진 질문
      "history" : st.session_state.messages
    }

    with st.chat_message("assistant"):
      with st.spinner('답변 생성중'):
        response = call_api(st.secrets['chatbot_url'],data)
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
      emotion = '{"emotion1":["위축감", 0.78], "emotion2":["억울함",0.11], "emotion3":["시기심",0.01]}' # 추후에 구현
      word = '["안녕","반갑","행복","안녕","외식","즐겁","안녕","고기","사랑","행복","가족"]' # 추후에 구현
      c.execute('INSERT INTO diarytable(diary_id, id, date, content, summary, emotion, word) VALUES (?,?,?,?,?,?,?)',(f"{datetime.today().strftime('%y%m%d')}_{st.session_state['id']}",
                                                                                                                      st.session_state['id'],
                                                                                                                      today,
                                                                                                                      str(st.session_state['messages']).replace("'", '"'),
                                                                                                                      summary,
                                                                                                                      emotion,
                                                                                                                      word))
      diary.commit()
      st.session_state['today_data'] = pd.DataFrame(data=[[f"{datetime.today().strftime('%y%m%d')}_{st.session_state['id']}",
                                                          st.session_state['id'],
                                                          today,
                                                          str(st.session_state['messages']).replace("'", '"'),
                                                          summary,
                                                          emotion,
                                                          word]],
                                                    columns=st.session_state['my_data'].columns.to_list())
      st.switch_page('pages/summary.py')
          
          
if __name__ == '__main__':
  diary = sqlite3.connect('diary.db')
  c = diary.cursor()

  main()