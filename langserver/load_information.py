import chromadb
from langchain_community.document_loaders import TextLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from sentence_transformers import SentenceTransformer
import json
import os
from langchain_community.embeddings import OpenAIEmbeddings
import shutil

# shutil.rmtree(f"{os.getcwd()}/chroma_data")

def load_and_split_md(file_path):

    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
        ("####", "header4"),
        ("#####", "header5"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    header_splits = header_splitter.split_text(markdown_text)

    processed_splits = []
    for doc in header_splits:
        header = None
        for level in ['header5', 'header4', 'header3', 'header2', 'header1']:
            if level in doc.metadata and doc.metadata[level]:
                header = doc.metadata[level]
                break
        
        new_content = f"{header}\n{doc.page_content}"

        processed_splits.append({
            "content": new_content,
            "metadata": doc.metadata
        })

    return processed_splits

def search_documents(vectordb, query, k=3):
    results = vectordb.similarity_search_with_score(query, k=k)

    for doc, score in results:
        print(f"\nScore: {score}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content}")
        print("-" * 50)

def create_vectordb(splits, embeddings):
    vectordb = Chroma.from_texts(
        texts = [split["content"] for split in splits],
        embedding = embeddings,
        metadatas = [split["metadata"] for split in splits],
        persist_directory=f"{os.getcwd()}/chroma_data"
    )

    return vectordb

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings

def enhance_query(query):
    query_terms = {
        "시스템 상태":  ["시스템 상태", "시스템 모니터링", "상태 확인", "시스템 리소스",
        "CPU", "RAM", "메모리", "디스크", "사용량"]
    }

    enhanced_terms = []
    for key, terms in query_terms.items():
        if key in query:
            enhanced_terms.extend(terms)
    
    return " ".join([query] + enhanced_terms)


mark_down_path = os.getcwd() + "/data/platform_information_rag2.md"

splits = load_and_split_md(mark_down_path)

embeddings = create_embeddings()

vectordb = create_vectordb(splits, embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# result = retriever.invoke("시스템 상태 보는 방법")
# print(result)
# formatted_result = format_docs(result)
# print(formatted_result)