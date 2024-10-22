import os
import sys

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from load_model import model
from langchain.prompts import ChatPromptTemplate

import streamlit as st
from langserver.prompt import rag_chain, split_output, split_brackets
from langserver.utils import print_message, process_text
from PIL import Image

img_path = r"D:\projects\MOAI_chatbot\MOAI_chatbot_이미지"

app = FastAPI()

st.set_page_config(page_title="MOAI chatbot", page_icon="🤖")
st.title("MOAI 챗봇")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

prompt = ChatPromptTemplate.from_template("Tell me a joke")
chain = prompt | model

# Edit this to add the chain you want to add
add_routes(app, chain, path="/chat")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
