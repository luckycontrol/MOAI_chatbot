import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langserver.prompt import rag_chain, split_output, split_brackets
from langserver.utils import print_message, process_text
from PIL import Image

img_path = os.path.join(os.getcwd(), "MOAI_chatbot_이미지")

st.set_page_config(page_title="MOAI chatbot", page_icon="🤖")
st.title("MOAI 챗봇")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화를 출력해주는 코드
# print_message()

if user_input := st.chat_input("메시지를 입력하세요"):
    print(f'\n질문: {user_input}', '\n')
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(("user", user_input))
    with st.spinner("전송중..."):
        msg = rag_chain.invoke(user_input)
        split_result = split_output(msg)
        final_result = split_brackets(split_result)

        print(msg, '\n', '*' * 60)

        with st.chat_message("MOAI"):
            for item in final_result:
                if item.startswith("[") and item.endswith("]"):
                    img_name = f'{os.path.join(img_path, item)}.png'
                    image = Image.open(img_name)
                    st.image(image)
                else:
                    st.write(item)
