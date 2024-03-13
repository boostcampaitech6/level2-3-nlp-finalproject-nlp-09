import streamlit as st

st.set_page_config(
    page_title = "login",
    page_icon= "👋"
)

st.title("Login")

# 사이드바 설정
st.sidebar.header("login page")


id = st.text_input(label = "ID", value = "")
password = st.text_input(label = "password", value = "")

st.session_state["ID"] = id
st.session_state['Password'] = password

if st.button("login"):
    con = st.container()
    con.caption("Result")
    con.write("ID가 설정되었습니다.")