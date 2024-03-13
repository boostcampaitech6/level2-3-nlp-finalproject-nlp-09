from openai import OpenAI
import streamlit as st

openai_api_key = "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"

st.set_page_config(
    page_title = "ChatBot",
    page_icon= "💬"
)

st.header("💬 3월 00일 당신의 하루는 어땠나요?(Demo) 🤗")
st.caption("🚀 당신의 하루를 들려주세요..")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "오늘 하루는 어땠나요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo-0613", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)