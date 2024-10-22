import chromadb
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
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
    metadata.update(record.get("content", {}))
    for key, value in metadata.items():
        if isinstance(value, list):
            metadata[key] = ', '.join(map(str, value))
    return metadata

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# example 데이터 로드
examples = json.load(open(os.getcwd() + "/data/platform_few_shot.json", "r", encoding="utf-8"))

# example_selector = SemanticSimilarityExampleSelector.from_examples(
#     examples,
#     embeddings,
#     Chroma,

# )

# RAG 데이터 로드
loader = JSONLoader(
    file_path=os.getcwd() + "/data/platform_information_rag.json",
    jq_schema='.[]',
    content_key='content'
)
pages = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=0
# )

chroma_client = chromadb.PersistentClient(path=os.getcwd() + "/chroma_data")

vectorstore = Chroma.from_documents(pages, embeddings, client=chroma_client)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

result = retriever.invoke("Vision S/W 와 연결된 Camera 연결 상태")
formatted_result = format_docs(result)
# print(formatted_result)