from collections import Counter
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import json

from menu import menu_with_redirect

def main():
  menu_with_redirect()

  st.title(st.session_state['id'] + '님의 감정 분석')
  
  if len(st.session_state['my_data']) < 5:
    st.write(f'{st.session_state['id']}님의 대화 데이터가 부족하여 분석할 수가 없습니다. 최소 5회 이상의 대화가 필요합니다.')
  else:
    st.divider()

    st.header(f'{st.session_state['id']}님의 지난 감정')
    st.write(f'{st.session_state['id']}님의 지난 5일간의 감정 흐름은 다음과 같습니다.')
    
    data = st.session_state['my_data'].tail(5)

    emotion = data['emotion']
    negative = ['억울함', '외로움', '후회', '실망', '허망', '그리움', '수치심', '고통', '절망', '무기력', '아픔', '위축감', '놀람', '공포', '걱정', '초조함', '원망', '불쾌', '날카로움', '타오름',
                '반감', '경멸', '비위상함', '치사함', '불신감', '시기심', '외면', '냉담', '불만', '갈등', '답답함', '아쉬움', '답답함', '불편함', '난처함', '서먹함', '심심함', '싫증', '부끄러움',
                '죄책감', '미안함', '슬픔']
    value_list = []
    for e in emotion:
      title = json.loads(e)['emotion1'][0]
      value = json.loads(e)['emotion1'][1]

      if title in negative:
        value *= -1
      
      value_list.append(value)

    fig, ax = plt.subplots()
    ax.set_title(f'{st.session_state['id']}님의 감정 흐름')
    ax.plot(data['date'], value_list)
    st.pyplot(fig)

    st.header(f'{st.session_state['id']}님의 단어 사용 빈도')
    st.write(f'{st.session_state['id']}님이 가장 자주 사용한 단어는 다음과 같습니다.')
    col1, col2 = st.columns(2)

    with col1:
      word_list = []
      for word in data['word'].values:
        word_list += json.loads(word)
      word_counter = Counter(word_list)
      most_5 = word_counter.most_common(5)  
      word_title = [w[0] for w in most_5]
      word_value = [w[1] for w in most_5]
      fig, ax = plt.subplots()
      ax.bar(word_title, word_value)
      st.pyplot(fig)
    
    with col2:
      st.write(f'{st.session_state['id']}님은 지난 5일 동안')
      for word in most_5:
        st.write(f'{word[0]} {word[1]}회')
      st.write('를 사용하셨습니다.')
if __name__ == '__main__':
  main()