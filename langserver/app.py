from langchain_community.llms import Ollama
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain_core.prompts import PromptTemplate
import json
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate
import chromadb
from chromadb.config import Settings

loader = TextLoader(r"E:\Dev\projects\MOAI_chatbot\data\platform_information.txt")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

chroma_client = chromadb.PersistentClient(path="E:\Dev\projects\MOAI_chatbot\chroma_data")

print(chroma_client.heartbeat())

vectorstore = Chroma.from_documents(docs, embeddings, client=chroma_client)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

template = '''
<context>{context}</context>
<context> 의 내용을 바탕으로 질문에 답변해주세요. 간결하게 답변해주세요. 질문과 관련된 답변 외의 내용은 절대 답변하지 마세요.

{question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

model = Ollama(
    model="llama3-ko",
    temperature=0
)

st.title("MOAI Chatbot")

st.write("안녕하세요. 저는 MOAI 챗봇입니다. 무엇을 도와드릴까요?")

user_input = st.text_input("사용자 입력")

if st.button("전송"):
    if user_input:
        with st.spinner("전송중..."):
            rag_chain = (
                {'context': retriever | format_docs, 'question': RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )

            st.write(rag_chain.invoke(user_input))
