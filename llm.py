import os

from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_zhipuai_dev.chat import ChatZhipuAI
from pydantic import SecretStr
from zhipuai import ZhipuAI

from zhipuai_embedding import ZhipuAIEmbeddings

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")


def get_embedding():
    # Google Generative AI Embeddings
    # return GoogleGenerativeAIEmbeddings(
    #     google_api_key=SecretStr(GOOGLE_API_KEY) if GOOGLE_API_KEY else None,
    #     model="models/gemini-embedding-exp-03-07",  # models/text-embedding-004
    # )

    # ZhipuAI Embeddings
    zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY")
    return ZhipuAIEmbeddings(
        api_key=SecretStr(zhipuai_api_key) if zhipuai_api_key else None,
        model="embedding-3",
    )


def get_chat_model():
    # Google Generative AI Chat Model
    # return ChatGoogleGenerativeAI(
    #     google_api_key=SecretStr(GOOGLE_API_KEY) if GOOGLE_API_KEY else None,
    #     model="gemini-2.0-flash",
    #     temperature=0,
    # )

    # ZhipuAI Chat Model，不支持 RunnableParallel 运算
    return ChatZhipuAI(
        api_key=os.getenv("ZHIPUAI_API_KEY"), model="glm-4", temperature=0
    )
