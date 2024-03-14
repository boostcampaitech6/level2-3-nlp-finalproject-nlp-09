import hashlib
import time

import streamlit as st
import sqlite3
from openai import OpenAI

from chatbot import chatbot

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def join_user(username):
  c.execute(f'SELECT * FROM userstable WHERE username = ?', (username,))
  data = c.fetchall()
  return data

def what_is_ed():
  st.title('감정일기')
  st.write('감정일기는 감정이 어쩌구저쩌구 일기가 어쩌구저쩌구 챗봇이 어쩌구저쩌구')

def login():
  sidebar_title = st.sidebar.header('로그인')
  username = st.sidebar.text_input("ID")
  password = st.sidebar.text_input("Password",type='password')
  login = st.sidebar.button('로그인')
  signin = st.sidebar.button('회원가입')

  if login:
    # if password == '12345':
    create_usertable()
    hashed_pswd = make_hashes(password)
    result = login_user(username,check_hashes(password,hashed_pswd))

    if result:
      st.success("Logged In as {}".format(username))
      st.session_state['is_login'] = True
      st.rerun()
    else:
      st.sidebar.warning("아이디 혹은 비밀번호가 틀렸습니다.")

  if signin:
    result = join_user(username)
    if result:
      st.sidebar.error('이미 존재하는 아이디입니다.')
    else:
      create_usertable()
      add_userdata(username,make_hashes(password))
      conn.commit()
      st.sidebar.success(f'가입을 환영합니다 {username}')
      time.sleep(2)
      st.rerun()
      
def mainpage():
  st.sidebar.write('일기')
  st.sidebar.write('마이페이지')

  st.title('오늘의 일기')
  st.divider()

  if "count" not in st.session_state:
    st.session_state.count = 0

  if "messages" not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])
     
  if prompt := st.chat_input():
    st.session_state.count += 1
    st.session_state.messages.append({'generation_id': st.session_state.count, 'role': 'user', 'content': prompt})
    st.chat_message("user").markdown(prompt)

    data = {
      "generation_id": st.session_state.count, # generation task id
      "query" : prompt, # 주어진 질문
    }

    with st.chat_message("assistant"):
      with st.spinner('답변 생성중'):
        response = chatbot(data)
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
          time.sleep(2)
          st.markdown('요약띠')

      
      

def main():
  if "is_login" not in st.session_state:
    st.session_state['is_login'] = False
  
  if not st.session_state['is_login']:
    what_is_ed()
    login()
  else:
    mainpage()


if __name__ == '__main__':
  conn = sqlite3.connect('data.db')
  c = conn.cursor()
  main()