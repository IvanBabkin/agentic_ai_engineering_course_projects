from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from typing import Type
from pydantic import BaseModel, Field

class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query")

class OneTimeSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information. Can only be used ONCE per task."
    args_schema: Type[BaseModel] = SearchToolInput
    
    # Define as proper Pydantic fields
    serper_tool: SerperDevTool = Field(default_factory=SerperDevTool, exclude=True)
    used: bool = Field(default=False, exclude=True)
    
    def _run(self, query: str) -> str:
        if self.used:
            return "Search already used in this task. Please proceed with available information."
        
        self.used = True
        return self.serper_tool.run(query)