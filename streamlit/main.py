import hashlib
import time
import pandas as pd
from datetime import datetime
from PIL import Image

import streamlit as st
import sqlite3

from menu import menu
from login import *

def login():
  sidebar_title = st.sidebar.header('로그인')
  username = st.sidebar.text_input("ID")
  password = st.sidebar.text_input("Password",type='password')
  login = st.sidebar.button('로그인')
  signin = st.sidebar.button('회원가입')

  if login:
    create_usertable(user_c)
    create_diarytable(diary_c)
    hashed_pswd = make_hashes(password)
    result = login_user(user_c, username,check_hashes(password,hashed_pswd))

    if result:
      st.session_state['is_login'] = True
      st.session_state['id'] = username
      st.session_state['my_data'] = load_user_data(diary_c, username)
      st.switch_page('pages/diary.py')
    else:
      st.sidebar.warning("아이디 혹은 비밀번호가 틀렸습니다.")

  if signin:
    create_usertable(user_c)
    if not password:
        st.sidebar.error('비밀번호를 입력해주세요')
        return
    result = join_user(user_c, username)
    if result:
      st.sidebar.error('이미 존재하는 아이디입니다.')
    else:
      add_userdata(user_c, user_db, username,make_hashes(password))
      user_db.commit()
      st.sidebar.success(f'가입을 환영합니다 {username}님')
      st.session_state['is_login'] = True
      st.session_state['id'] = username
      create_diarytable(diary_c)
      st.session_state['my_data'] = load_user_data(diary_c, username)
      time.sleep(2)
      st.switch_page('pages/diary.py')

def what_is_ed():
  st.title('A BoostCamp Diary for Emotions')
  st.write('하루의 일상을 마무리하면서 Dr.부덕이와 나눈 대화를 바탕으로 일기를 생성해주는 감정 일기 서비스')
  st.subheader('오늘의 일기를 쓰기 전에 부덕이가 할말이 있데요!')
  st.image('images/howto.png')
  st.write('대화를 최소 4번 이상 주고받아야 Dr.부덕이가 일기를 요약할 수 있어!')

def main():
  for key in st.session_state.keys():
    del st.session_state[key]
    
  st.session_state['today'] = datetime.now().date()
  st.session_state['is_login'] = False
  st.session_state['my_data'] = ''
  st.session_state['id'] = ''

  menu()
  login()
  what_is_ed()
  st.sidebar.page_link('https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-09', label='Visit our Github')

if __name__ == '__main__':
  user_db = sqlite3.connect('user_data.db')
  diary = sqlite3.connect('diary.db')
  user_c = user_db.cursor()
  diary_c = diary.cursor()
  main()