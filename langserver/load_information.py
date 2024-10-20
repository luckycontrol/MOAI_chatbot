import chromadb
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import os

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata
        formatted_doc = f"컨텐츠:\n{content}\n\n메타데이터:\n"
        for key, value in metadata.items():
            formatted_doc += f"{key}: {value}\n"
        formatted_docs.append(formatted_doc)
    return '\n\n---\n\n'.join(formatted_docs)

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata.update(record.get("metadata", {}))
    for key, value in metadata.items():
        if isinstance(value, list):
            metadata[key] = ', '.join(map(str, value))
    return metadata

# example 데이터 로드
examples = json.load(open(os.getcwd() + "/data/platform_few_shot.json", "r", encoding="utf-8"))

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True}
)

# RAG 데이터 로드
# loader = TextLoader(os.getcwd() + "/data/platform_information.txt")
loader = JSONLoader(
    file_path=os.getcwd() + "/data/platform_information_rag.json",
    jq_schema='.[]',
    content_key='content',
    metadata_func=metadata_func
)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150
)

# docs = text_splitter.split_documents(pages)

chroma_client = chromadb.PersistentClient(path=os.getcwd() + "/chroma_data")

vectorstore = Chroma.from_documents(pages, embeddings, client=chroma_client)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# result = retriever.invoke("PLC 와 연결된 상태를 확인하는 방법을 알려줘.")
# formatted_result = format_docs(result)
# print(formatted_result)