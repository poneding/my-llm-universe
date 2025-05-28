import os
import re

from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from llm import get_embedding

_ = load_dotenv(find_dotenv())


file_paths = []
for root, dirs, files in os.walk("knowledge_base"):
    for file_path in files:
        file_paths.append(os.path.join(root, file_path))

loaders = []
for file_path in file_paths:
    file_type = file_path.split(".")[-1]
    if file_type == "pdf":
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == "md":
        loaders.append(UnstructuredMarkdownLoader(file_path))

texts = []
for loader in loaders:
    texts.extend(loader.load())

# 切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(texts)


vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=get_embedding(),
    persist_directory="vector_db",
)
