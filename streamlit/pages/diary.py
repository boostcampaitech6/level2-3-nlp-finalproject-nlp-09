import time

import streamlit as st

from menu import menu_with_redirect
from chatbot import *



def main():
  menu_with_redirect()

  if "diary_done" not in st.session_state:
      st.session_state['diary_done'] = True

  st.title('오늘의 일기')
  st.divider()

  if "count" not in st.session_state:
    st.session_state.count = 0

  if "messages" not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])

  if st.session_state['diary_done']:
    prompt = st.chat_input(disabled=True)
  else:
    prompt = st.chat_input()

  if prompt:
    st.session_state.count += 1
    st.session_state.messages.append({'generation_id': st.session_state.count, 'role': 'user', 'content': prompt})
    st.chat_message("user").markdown(prompt)

    data = {
      "generation_id": st.session_state.count, # generation task id
      "query" : prompt, # 주어진 질문
    }

    with st.chat_message("assistant"):
      with st.spinner('답변 생성중'):
        response = call_api(data)
      st.session_state.count += 1
      st.session_state.messages.append({'generation_id': st.session_state.count, "role": "assistant", "content": response})
      st.markdown(response)

  if st.session_state.count == 10:
    with st.chat_message("assistant"):
      st.markdown('내용을 요약하시겠습니까?')
      summary_btn = st.button('요약하기')
    if summary_btn:
      with st.chat_message("assistant"):
        with st.spinner('요약중'):
          data = {
            "generation_id": st.session_state.count, # generation task id
            "query" : '지금까지 한 대화를 한 줄로 요약해줘', # 주어진 질문
          }
          response = call_api(data)
      st.markdown(response)
          

if __name__ == '__main__':
  main()