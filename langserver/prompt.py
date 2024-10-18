import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langserver.load_model import model
from langserver.load_information import retriever, examples

example_prompt = PromptTemplate.from_template(
    "question: {question}\nanswer: {answer}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="다음은 질문과 답변의 예시입니다:",
    suffix="이제 다음 질문에 답해주세요:\n질문: {question}\n답변:",
    input_variables=["question"]
)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

template = '''
{few_shot_examples}

<CONTEXT>{context}</CONTEXT>

<CONTEXT>의 내용을 바탕으로 질문에 대해서만 간결하게 답변하세요.
만약 <CONTEXT>에 질문과 관련된 정보가 없다면 '제공된 텍스트에는 해당 질문에 대한 정보가 포함되어 있지 않습니다.'라고 응답하세요.

답변 시 중복된 내용을 피하고 사용자가 간결한 답을 요청하기 전까지는 완전하고 상세한 답변을 제공하세요.
동일한 정보를 반복하지 말고, 한 번만 언급하세요.
정보를 요약할 때는 중복을 제거하고 핵심 내용만 포함하세요.

질문: {question}
답변:
'''

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {
        "few_shot_examples": few_shot_prompt,
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

print(rag_chain.invoke("Master 와 Slave 가 연결되었다면 어떻게 표시되나요?"))
