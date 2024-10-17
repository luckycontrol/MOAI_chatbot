import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langserver.load_information import docs
from langserver.prompt import rag_chain
from langserver.load_model import model
from langserver.utils import print_message

st.set_page_config(page_title="MOAI chatbot", page_icon="🤖")
st.title("MOAI 챗봇")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화를 출력해주는 코드
print_message()

if user_input := st.chat_input("메시지를 입력하세요"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(("user", user_input))
    with st.spinner("전송중..."):
        msg = rag_chain.invoke(user_input)
        st.chat_message("MOAI").write(msg)
        st.session_state["messages"].append(("MOAI", msg))
