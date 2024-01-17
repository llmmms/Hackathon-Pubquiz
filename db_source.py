import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from prompts import pubquiz_prompt
from llm import llm
from db import db


load_dotenv(override=True)
azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')


def get_retrieval_chain(db, llm):
    document_prompt = ChatPromptTemplate.from_template("""Content: {page_content}""")
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=pubquiz_prompt,
        document_prompt=document_prompt,
    )

    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
