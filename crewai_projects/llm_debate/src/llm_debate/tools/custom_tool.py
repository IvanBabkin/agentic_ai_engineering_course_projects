from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from typing import Type
from pydantic import BaseModel, Field


class SearchToolInput(BaseModel):
    """Schema for the web-search tool."""
    query: str = Field(..., description="Search query")


class OneTimeSearchTool(BaseTool):
    """
    Web-search tool that can be used once per task execution.
    """
    name: str = "web_search_once"
    description: str = "Search the web. Can only be invoked once per task."
    args_schema: Type[BaseModel] = SearchToolInput

    def _run(self, query: str, **_ignored) -> str:
        """Perform the web search."""
        serper_tool = SerperDevTool(
            n_results=5
        )
        result = serper_tool.run(
            search_query=query)
        return f"""{result} 
            \n\nweb search tool was now called for this task, 
            no further web searches available for this task"""