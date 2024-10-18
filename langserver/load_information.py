import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

loader = TextLoader(r"D:\projects\MOAI_chatbot\data\platform_information.txt")
pages = loader.load()

examples = json.load(open(r"D:\projects\MOAI_chatbot\data\platform_few_shot.json", "r", encoding="utf-8"))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(pages)

chroma_client = chromadb.PersistentClient(path="D:\projects\MOAI_chatbot\chroma_data")

vectorstore = Chroma.from_documents(docs, embeddings, client=chroma_client)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})