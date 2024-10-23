import chromadb
from langchain_community.document_loaders import TextLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from sentence_transformers import SentenceTransformer
import json
import os
# import shutil

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata
        formatted_doc = f"컨텍스트: {content}"
        
        formatted_docs.append(formatted_doc)
        
    return ('\n\n---\n\n'.join(formatted_docs), metadata.get("image_ref", None))

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata.update(record.get("metadata", {}))
    for key, value in metadata.items():
        if isinstance(value, list):
            metadata[key] = ', '.join(map(str, value))
    return metadata

def load_and_split_md(file_path):

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    header_splits = header_splitter.split_text(markdown_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(header_splits)
    return chunks

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": False}
)

# RAG 데이터 로드
chunks = load_and_split_md(os.getcwd() + "/data/platform_information_rag2.md")

chroma_client = chromadb.PersistentClient(path=os.getcwd() + "/chroma_data")

vectorstore = Chroma.from_documents(
    chunks, 
    embeddings, 
    client=chroma_client
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,  # 상위 4개 문서만 반환
    }
)

result = retriever.invoke("시스템 상태 보는 방법")
print(result)
# formatted_result = format_docs(result)
# print(formatted_result)