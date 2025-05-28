import os

from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from llm import get_chat_model, get_embedding

_ = load_dotenv(find_dotenv())


vector_store = Chroma(
    embedding_function=get_embedding(),
    persist_directory="vector_db",
)

llm = get_chat_model()

# template v1
# template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
# {context}
# 问题: {question}
# """

# template v2
# template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
# 案。你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。
# {context}
# 问题: {question}
# 有用的回答:"""

# template v3
# template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
# 案。你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。
# 如果答案有几点，你应该分点标号回答，让答案清晰具体
# {context}
# 问题: {question}
# 有用的回答:"""

# template v4
# template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
# 案。你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。
# 如果答案有几点，你应该分点标号回答，让答案清晰具体。
# 请你附上回答的来源原文，以保证回答的正确性。
# {context}
# 问题: {question}
# 有用的回答:"""

template = """
请你依次执行以下步骤：
① 使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。
如果答案有几点，你应该分点标号回答，让答案清晰具体。
上下文：
{context}
问题: 
{question}
有用的回答:
② 基于提供的上下文，反思回答中有没有不正确或不是基于上下文得到的内容，如果有，回答你不知道
确保你执行了每一个步骤，不要跳过任意一个步骤。
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template=template)


def combine_docs(docs) -> str:
    """Combine the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


retrieval_chain = vector_store.as_retriever() | RunnableLambda(combine_docs)

qa_chain = (
    RunnableParallel(context=retrieval_chain, question=RunnablePassthrough())
    | QA_CHAIN_PROMPT
    | llm
    | StrOutputParser()
)

# question = "什么是南瓜书？"
# result = qa_chain.invoke(question)
# print(result)

# question = "使用大模型时，构造 Prompt 的原则有哪些"
# result = qa_chain.invoke(question)
# print(result)

# question = "强化学习的定义是什么"
# result = qa_chain.invoke(question)
# print(result)

# question = "我们应该如何去构造一个 LLM 项目"
# result = qa_chain.invoke(question)
# print(result)

# question = "LLM 的分类是什么？给我返回一个 Python List"
# result = qa_chain.invoke(question)
# print(result)


# 设置一个 LLM（agent）来理解指令
def gen_prompt(input: str) -> list[dict]:
    """
    生成 LLM 的输入提示。
    :param input: 用户输入的指令
    :return: LLM 输入提示的列表
    """
    return [{"role": "user", "content": input}]


def get_completion(prompt: str):
    client = get_chat_model()
    response = client.invoke(prompt)
    return response.content


prompt_input = """
请判断以下问题中是否包含对输出的格式要求，并按以下要求输出：
请返回给我一个可解析的Python列表，列表第一个元素是对输出的格式要求，应该是一个指令；第二个元素是去掉格式要求的问题原文
如果没有格式要求，请将第一个元素置为空
需要判断的问题：
```
{}
```
不要输出任何其他内容或格式，确保返回结果可解析。
"""

question = "LLM 的分类是什么？给我返回一个 Python List"
response = get_completion(prompt_input.format(question))
print(response)
