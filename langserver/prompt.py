import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate, PipelinePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langserver.load_model import model
from langserver.load_information import retriever, examples, format_docs

example_prompt = PromptTemplate.from_template(
    "question: {question}\nanswer: {answer}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix='다음은 질문과 답변의 예시입니다.',
    suffix='''
    <example>   
    {few_shot_examples}
    </example>

    위의 예시와 같이 답변하세요.
    image_ref 를 [ ] 로 표시하여 예시와 같이 답변하세요
    예: [공통UI_003] 위 이미지에서 PC 로고를 통해서 PC 연결 상태를 알 수 있습니다.
    ''',
    input_variables=["question"]
)

final_prompt = '''
{context}

질문에 대해 다음 지침을 따라 답변해 주세요:
1. 간결하고 정확한 답변을 제공해 주세요.
2. 컨텍스트에 제공된 정보만을 사용해 주세요.
3. [공통UI_XXX]와 같은 이미지 참조를 포함하여 설명해 주세요.
4. 불필요한 정보는 제외하고 질문과 직접적으로 관련된 내용만 답변해 주세요.
5. 확실하지 않은 정보에 대해서는 추측하지 마세요.

질문: {question}
답변:
'''

prompt = ChatPromptTemplate.from_template(final_prompt)

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

print(rag_chain.invoke("카메라의 연결상태를 확인하는 방법을 알려줘."))