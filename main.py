import os
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.vectorstores.chroma import Chroma
from tools import wiki_tool, wolframalpha_tool, ddg_tool, db_tool, google_tool

from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()

# LLM
azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

llm = AzureChatOpenAI(
    api_key=azure_api_key,
    api_version="2023-05-15",
    azure_deployment="gpt-35-turbo-16k",
    azure_endpoint=azure_endpoint,
)

llm.invoke("What do you know about Pub Quizzes?")

# Chroma db
embeddings = AzureOpenAIEmbeddings(
    api_key=azure_api_key,
    api_version="2023-05-15",
    azure_deployment="text-embedding-ada-002",
    azure_endpoint=azure_endpoint,
)

db = Chroma(persist_directory="./PubDatabase/chroma", embedding_function=embeddings)

# Example Agent

qa_tool = Tool.from_function(
    func=llm.invoke,
    name="QA",
    description="Tool to answer a question",
)

PREFIX = """You are participating in a pubquiz. Answer in a short sentence."""
agent = initialize_agent(
    tools=[qa_tool, wiki_tool, ddg_tool, wolframalpha_tool, db_tool, google_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    agent_kwargs={
        'prefix': PREFIX
    }
)

with open("PubTexts/TypischeFragen.txt") as f:
    example_questions = f.read()

intent_prompt = ChatPromptTemplate.from_template(
"""
Consider the following examples:
"""
+ example_questions
+ """Now give a short answer the following Question.
Question: {input}""")

intent_chain = intent_prompt | llm

# user_input = input("What do you know about Pub Quizzes")
user_input = input("Bitte Frage eingeben / Please enter your question: ")

agent.invoke({"input": user_input})

# whisper

azure_api_key_whisper = os.getenv('AZURE_OPENAI_API_KEY_WHISPER')
azure_endpoint_whisper = os.getenv('AZURE_OPENAI_ENDPOINT_WHISPER')

whisper = AzureOpenAI(
    api_key=azure_api_key_whisper,
    azure_endpoint=azure_endpoint_whisper,
    azure_deployment="whisper",
    api_version="2023-09-01-preview",
)

# user_input = input("What do you know about Pub Quizzes")
user_input = input("Bitte Frage eingeben / Please enter your question: ")

agent.invoke({"input": user_input})
