from dotenv import load_dotenv

from langchain.tools import Tool
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper

load_dotenv()

ddg = DuckDuckGoSearchRun()
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
