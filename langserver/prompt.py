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
<|start_header_id|>
{few_shot_examples}

간결하고 짧은 답변을 제공하세요. 중복되지 않게 답변하세요.
<|end_header_id|>

<s>{question}
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

print(rag_chain.invoke("Vision S/W 의 상단 공통 UI 에 대해 설명해줘"))
