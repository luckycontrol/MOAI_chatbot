from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .load_model import model
from .load_information import retriever

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

template = '''
<context>{context}</context>
<context> 의 내용을 바탕으로 질문에 간결하게 답변하세요.
<context> 의 내용에서 질문에 대한 내용이 없을 경우, '제공된 텍스트에는 질문에 대한 정보가 없습니다' 라고 답변하세요.

{question}
'''

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)