from datetime import datetime
import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import Counter


import streamlit as st

from menu import menu_with_redirect
from chatbot import *

def main():
  menu_with_redirect()
  today = datetime.now().date()
  today_data = st.session_state['my_data'][st.session_state['my_data']['date']==str(today)]

  st.title(f'{datetime.now().year}년 {datetime.now().month}월 {datetime.now().day}일 오늘의 일기')
  st.divider()

  if today_data.empty:
    st.write('오늘의 일기가 없습니다. 오늘의 일기를 먼저 작성해주세요.')
  else:
    st.header('오늘의 일기 요약')
    st.write(today_data['summary'].values[0])

    col1, col2 = st.columns(2)

    with col1:
      st.header('오늘의 감정 퍼센트')
      emotion = json.loads(today_data['emotion'].values[0])
      emotion_title = [data[0] for data in emotion.values()]
      emotion_value = [data[1] for data in emotion.values()]
      fig, ax = plt.subplots()
      ax.pie(emotion_value, labels=emotion_title, autopct='%1.1f%%')
      st.pyplot(fig)
      
    with col2:
      st.header('오늘의 단어 퍼센트')
      word_list = json.loads(today_data['word'].values[0])
      word_list = Counter(word_list)
      fig, ax = plt.subplots()
      ax.bar(word_list.keys(), word_list.values())

      st.pyplot(fig)

if __name__ == '__main__':
  main()