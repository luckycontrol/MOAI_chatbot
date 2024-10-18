import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langserver.prompt import rag_chain
from langserver.utils import print_message, process_text
from PIL import Image

img_path = r"D:\projects\MOAI_chatbot\MOAI_chatbot_이미지"

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
        result = process_text(msg)

        for item in result:
            if item.startswith("이미지:"):
                img_name = f'{item.split(":")[1].strip()}.png'
                image_path = os.path.join(img_path, img_name)
                image = Image.open(image_path)
                st.image(image, caption=img_name)
                st.session_state["messages"].append(("MOAI", image))
            else:
                text = item.split(":")[1].strip()
                st.chat_message("MOAI").write(text)
                st.session_state["messages"].append(("MOAI", text))