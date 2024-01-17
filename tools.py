from langchain.tools import Tool
from langchain.tools.ddg_search import DuckDuckGoSearchRun

ddg = DuckDuckGoSearchRun()

ddg_tool = Tool.from_function(
    func = ddg.run,
    name = "DuckDuckGo Search",
    description = "Search DuckDuckGo for a query abount current events.",
)

