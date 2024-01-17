"""Only run this once to add content to the database."""

# TODO: Check for duplicate entries

import os
from pathlib import Path

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv
load_dotenv()


TO_ADD = [
    'texts',
    # 'audio',
]

# LLM
azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

# persist_directory = "./chroma/book"
persist_directory = "./PubDatabase/chroma"
text_dir = Path("./PubTexts/")


def get_text_data(text_dir):
    text_data = []
    for text_file in text_dir.glob("*.txt"):
        loader = TextLoader(str(text_file), encoding="utf-8")
        text_data.extend(loader.load())
    return text_data

def get_text_documents(text_dir=text_dir):
    data = get_text_data(text_dir=text_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separators=[".", "\n"])
    documents = splitter.split_documents(data)
    return documents

embeddings = AzureOpenAIEmbeddings(
    api_key=azure_api_key,
    api_version="2023-05-15",
    azure_deployment="text-embedding-ada-002",
    azure_endpoint=azure_endpoint,
)

db = Chroma(persist_directory="./PubDatabase/chroma", embedding_function=embeddings)

if 'texts' in TO_ADD:
    db.add_documents(get_text_documents(text_dir=text_dir))

