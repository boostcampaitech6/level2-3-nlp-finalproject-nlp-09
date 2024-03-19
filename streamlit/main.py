import hashlib
import time
import pandas as pd
from datetime import datetime

import streamlit as st
import sqlite3
from openai import OpenAI

from menu import menu

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
	user_db.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def join_user(username):
  c.execute('SELECT * FROM userstable WHERE username = ?', (username,))
  data = c.fetchall()
  return data

def create_diarytable():
  diary_c.execute('CREATE TABLE IF NOT EXISTS diarytable (diary_id TEXT, id INTEGER, date DATE, content TEXT, summary TEXT, emotion TEXT, word TEXT, PRIMARY KEY (diary_id, id))')

def load_user_data(username):
  diary_c.execute('SELECT * FROM diarytable WHERE id = ?', (username,))
  data = diary_c.fetchall()
  data = pd.DataFrame(data, columns=['diary_id', 'id', 'date', 'content', 'summary', 'emotion','word'])
  data = data.apply(lambda x:x.replace("'", '"'))
  return data

def login():
  sidebar_title = st.sidebar.header('로그인')
  username = st.sidebar.text_input("ID")
  password = st.sidebar.text_input("Password",type='password')
  login = st.sidebar.button('로그인')
  signin = st.sidebar.button('회원가입')

  if login:
    create_usertable()
    create_diarytable()
    hashed_pswd = make_hashes(password)
    result = login_user(username,check_hashes(password,hashed_pswd))

    if result:
      st.success("Logged In as {}".format(username))
      st.session_state['is_login'] = True
      st.session_state['id'] = username
      st.session_state['my_data'] = load_user_data(username)
      st.session_state['today_data'] = st.session_state['my_data'][st.session_state['my_data']['date']==str(today)]
      st.switch_page('pages/diary.py')
    else:
      st.sidebar.warning("아이디 혹은 비밀번호가 틀렸습니다.")

  if signin:
    result = join_user(username)
    if result:
      st.sidebar.error('이미 존재하는 아이디입니다.')
    else:
      create_usertable()
      add_userdata(username,make_hashes(password))
      user_db.commit()
      st.sidebar.success(f'가입을 환영합니다 {username}님')
      st.session_state['is_login'] = True
      st.session_state['id'] = username
      create_diarytable()
      st.session_state['my_data'] = load_user_data(username)
      time.sleep(2)
      st.switch_page('pages/diary.py')

def what_is_ed():
  st.title('A BoostCamp Diary for Emotions')
  st.write('하루의 일상을 마무리하면서 Dr.부덕이와 나눈 대화를 바탕으로 일기를 생성해주는 감정 일기 서비스')

def main():
  if "is_login" not in st.session_state:
    st.session_state['is_login'] = False

  if "my_data" not in st.session_state:
    st.session_state['my_data'] = None

  if "id" not in st.session_state:
    st.session_state['id'] = None

  if "today_data" not in st.session_state:
    st.session_state['today_data'] = None

  menu()
  login()
  what_is_ed()

if __name__ == '__main__':
  user_db = sqlite3.connect('user_data.db')
  diary = sqlite3.connect('diary.db')
  c = user_db.cursor()
  diary_c = diary.cursor()
  today = datetime.now().date()
  main()