from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI

model = Ollama(
    model="gemma2",
    temperature=0
)

# model = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0,
#     api_key="AIzaSyC1lzjbT1BFOx83dPHRJLT7mJjhcvbR6ZU"
# )