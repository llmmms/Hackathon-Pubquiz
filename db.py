import os
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

# LLM
azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

# Chroma db
embeddings = AzureOpenAIEmbeddings(
    api_key=azure_api_key,
    api_version="2023-05-15",
    azure_deployment="text-embedding-ada-002",
    azure_endpoint=azure_endpoint,
)

db = Chroma(persist_directory="./PubDatabase/chroma", embedding_function=embeddings)
