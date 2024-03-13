import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.header("사용자 감정 통계량")

st.sidebar.header("Result page")

# 결과화면 차트 예제

data = {
    'Year': [2010, 2011, 2012, 2013, 2014, 2015],
    'Sales': [100, 150, 200, 250, 300, 350]
}

df = pd.DataFrame(data)

# 데이터 테이블 출력
st.subheader("Data Table")
st.write(df)

# 그래프 출력
st.subheader("bar graph")
plt.bar(df['Year'],df['Sales'])
st.pyplot()

# 라인 그래프 출력
st.subheader("라인 그래프")
plt.plot(df['Year'],df['Sales'])
st.pyplot()