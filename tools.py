from dotenv import load_dotenv

from langchain.tools import Tool
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper

from db_source import invoke_db

load_dotenv()

ddg = DuckDuckGoSearchRun()
google = GoogleSearchAPIWrapper()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wolfram = WolframAlphaAPIWrapper()


ddg_tool = Tool.from_function(
    func = ddg.run,
    name = "DuckDuckGo Search",
    description = "Search DuckDuckGo for a query about current events.",
)

wiki_tool = Tool.from_function(
    func = wiki.run,
    name = "Wikipedia Query",
    description = "Search Wikipedia for queries on factual knowledge."
)

wolframalpha_tool = Tool.from_function(
    func=wolfram.run,
    name="WolframAlpha",
    description="Query WolframAlpha for mathematical or physical calculations."
)

db_tool = Tool.from_function(
    func = invoke_db,
    name = "Database Retrieval",
    description = "Use Database Retrieval for factual knowledge before all other tools. Do not use it for querying calculations!"
)

google_tool = Tool.from_function(
    func=google.run,
    name="Google Search",
    description="Do a Google Search for recent events or live information, such as weather or news. Give this tool higher preference than DuckDuckGo Search."
)
