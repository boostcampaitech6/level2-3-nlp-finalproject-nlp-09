import streamlit as st
import time, os
from app.models import ChatPipe

os.environ['HF_TOKEN'] = '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'

# pipe = ChatPipe(
#     "EleutherAI/polyglot-ko-5.8b", 
#     'm2af/EleutherAI-polyglot-ko-5.8b-adapter', 
#     streamer=True
# )

pipe = ChatPipe(
    "CurtisJeon/OrionStarAI-Orion-14B-Chat-4bit", 
    # '/home/jhw/level2-3-nlp-finalproject-nlp-09/models/chat/trained/Orion-Base-240323-043404/loaded', 
    streamer=False
)

# Streamed response emulator
def response_generator(text, history):
    if "요약" in text:
        prompt = pipe.summary_prompt(history)
        result = pipe.pipe(prompt)
        reply = pipe.post_process(result)
    else:
        reply = pipe(text, history)
    for word in reply.split():
        yield word + " "
        time.sleep(0.05)

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "오늘 하루 어땠어?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    print(st.session_state.messages)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, st.session_state.messages))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})