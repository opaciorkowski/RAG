import os
import tempfile
import urllib.request
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY_TEG")

BASE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"

llm = ChatOpenAI(api_key=openai_api_key, model=BASE_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=EMBEDDING_MODEL)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVER_K = 4

url = "https://www.nutrition.va.gov/docs/UpdatedPatientEd/2022CookingAroundtheWorldCookbook.pdf"
response = requests.get(url)
pdf_file = BytesIO(response.content)