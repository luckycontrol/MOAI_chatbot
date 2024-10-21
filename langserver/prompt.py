import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate, PipelinePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langserver.load_model import model
from langserver.load_information import retriever, examples, format_docs
import re

import re

def split_output(output):
    lines = output.split('\n')
    
    result = []
    seen = set()
    for line in lines:
        line = line.strip()
        if line and not line.startswith('</s>'):  # 빈 줄과 '</s>'로 시작하는 줄 제거
            # Extract the content inside brackets
            bracket_content = re.search(r'\[([^\]]+)\]', line)
            if bracket_content:
                bracket_key = bracket_content.group(0)
                if bracket_key not in seen:
                    seen.add(bracket_key)
                    if re.match(r'^[0-9]+\.', line):
                        # 번호가 매겨진 리스트 항목
                        result.append(line)
                    else:
                        # 일반 텍스트 줄
                        parts = re.split(r'[.:]', line)
                        parts = [part.strip() for part in parts if part.strip()]
                        result.extend(parts)
            else:
                if re.match(r'^[0-9]+\.', line):
                    # 번호가 매겨진 리스트 항목
                    result.append(line)
                else:
                    # 일반 텍스트 줄
                    parts = re.split(r'[.:]', line)
                    parts = [part.strip() for part in parts if part.strip()]
                    result.extend(parts)
    
    # 결과에서 '</s>'로 끝나는 항목 제거
    result = [item for item in result if not item.endswith('</s>')]
    
    return result

def split_brackets(current_result):
    output = []
    for item in current_result:
        if '[' in item and ']' in item:
            # [ ]로 둘러싸인 부분 분리
            parts = re.split(r'(\[[^]]+\])', item)
            for part in parts:
                if part.strip():
                    if part.startswith('[') and part.endswith(']'):
                        output.append(part)
                    else:
                        # 번호가 매겨진 리스트 항목 재구성
                        match = re.match(r'^(\d+\.\s*)(.*)', part)
                        if match:
                            number, content = match.groups()
                            output.append(number.strip() + ' ' + content.strip())
                        else:
                            output.append(part.strip())
        else:
            output.append(item)
    return output

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

    위의 예시의 답변과 같은 형식으로 답변하세요.
    반드시 제공된 컨텍스트 정보만을 사용하여 답변하세요.
    ''',
)

final_prompt = '''
주어진 컨텍스트:
{context}

질문에 대해 다음 지침을 엄격히 따라 답변해 주세요:
1. 오직 주어진 컨텍스트 정보만을 사용하여 답변하세요.
2. 컨텍스트의 'content' 와 'image_ref' 정보를 이용하여 문장을 자연스럽게 작성하세요.
3. 컨텍스트에 'image_ref' 가 있을 경우 반드시 참조하세요.
4. 컨텍스트의 'image_ref' 는 반드시 대괄호 안에 표시하세요. 그리고 다음과 같이 문장을 작성하세요. ex) [공통UI_003] 위 이미지의 PC 로고를 통해서 알 수 있습니다. [공통UI_004] 위 이미지는 Master 와 Slave 가 연결되었음을 의미하고, [공통UI_005] 위 이미지는 Master 와 Slave 가 연결되지 않았음을 의미합니다.
5. 답변 끝에는 마침표를 붙이세요.
6. 컨텍스트에 관련 정보가 없는 경우, "주어진 정보로는 답변할 수 없습니다."라고 명시하세요.

{question}
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

result = rag_chain.invoke("조명 값을 바꿀려면 어떻게 해야하나요?")
split_result = split_output(result)
# print(split_result)
print(split_brackets(split_result))
