from datetime import datetime
import pandas as pd
import json
from collections import Counter

import streamlit as st
import matplotlib.pyplot as plt
import koreanize_matplotlib

from menu import menu_with_redirect
from chatbot import *

def main():
  menu_with_redirect()
  today = datetime.now().date()
  
  st.title(f'{datetime.now().year}년 {datetime.now().month}월 {datetime.now().day}일 일기 요약')
  st.divider()

  if st.session_state['today_data'].empty:
    st.write('오늘의 일기가 없습니다. 오늘의 일기를 먼저 작성해주세요.')
  else:
    st.header('오늘의 일기 요약')
    st.write(st.session_state['today_data']['summary'].values[0])

    emotion = json.loads(st.session_state['today_data']['emotion'].values[0])
    emotion_title = [data[0] for data in emotion.values()]
    emotion_value = [data[1] for data in emotion.values()]

    st.header('오늘의 감정 퍼센트')
    st.write(f'오늘은 {st.session_state.id}님의 감정은 {emotion_title[0]} {emotion_value[0] * 100:.0f}%, {emotion_title[1]} {emotion_value[1] * 100:.0f}%, {emotion_title[2]} {emotion_value[2] * 100:.0f}%, 기타 {emotion_value[3] * 100:.0f}%입니다.')
    col1, col2 = st.columns(2)

    with col1:
      fig, ax = plt.subplots()
      ax.pie(emotion_value, labels=emotion_title, autopct='%1.1f%%')

      st.pyplot(fig)

if __name__ == '__main__':
  main()