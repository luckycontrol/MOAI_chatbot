import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate, PipelinePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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

# example_prompt = PromptTemplate.from_template(
#     "question: {question}\nanswer: {answer}"
# )

# few_shot_prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     prefix='다음은 질문과 답변의 예시입니다.'
# )

final_prompt = '''<start_of_turn>user
다음은 RAG 결과입니다:
{context}

위의 RAG 결과를 사용하여 다음 지침에 따라 답변하세요:
1. RAG 결과에서 metadata 의 image_ref 와 page_content 를 참조하여 답변하세요.
2. image_ref 는 이미지 참조입니다. 반드시 이미지 참조를 포함하여 답변하세요. 단, 이미지 참조: [메인UI_003] 와 같이 작성하지 마세요. 이미지를 확인하세요 또는 이미지에서 확인 가능합니다 와 같이 작성하세요.
3. 문장이 어색하지 않도록 정리하여 답변하세요.
4. 답변 내용이 중복되지 않도록 하세요.
5. ** ** 와 같은 강조 표현은 사용하지 마세요.

"question": {question}

<end_of_turn>

<start_of_turn>model
'''


prompt = ChatPromptTemplate.from_template(final_prompt)

question = "display 수량을 변경하는 방법"

# rag 결과 확인
# rag_result = retriever.invoke(question)
# print(rag_result)

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# rag 결과를 통한 답변 생성 확인
# result = rag_chain.invoke(question)
# split_result = split_output(result)
# print(split_result)
# print(split_brackets(split_result))
