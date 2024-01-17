from langchain.tools import Tool
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

ddg = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

ddg_tool = Tool.from_function(
    func = ddg.run,
    name = "DuckDuckGo Search",
    description = "Search DuckDuckGo for a query abount current events.",
)

wiki_tool = Tool.from_function(
    func = wiki.run,
    name = "Wikipedia Query",
    description = "Query Wikipedia for answer."
)

