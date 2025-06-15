from pydantic import BaseModel, Field
from agents import Agent

def generate_planner_agent_instructions(n_searches: int):
    INSTRUCTIONS = f"""
        You are an expert research strategist. Given a query (possibly with additional clarification context), create a comprehensive search plan with {n_searches} diverse and strategic web searches.

        When additional context is provided from clarification questions, use that information to:
        - Focus searches on the specific aspects the user is most interested in
        - Adjust the scope and depth based on their preferences
        - Target the appropriate audience level (technical vs. general)
        - Consider the specified time frame or geographical focus
        - Include the requested perspectives or viewpoints

        For each search, consider:
        - **Relevance to clarified intent**: Prioritize aspects highlighted in clarification
        - **Coverage**: Ensure searches cover different aspects/angles of the topic
        - **Specificity**: Balance broad overview searches with specific detailed searches  
        - **Recency**: Include searches for recent developments or current information
        - **Authority**: Consider searches that would find authoritative/expert sources
        - **Practical**: Include searches for real-world applications or examples

        Search types to consider:
        - Overview/definition searches
        - Expert opinion/analysis searches  
        - Recent news/developments searches
        - Statistical/data searches
        - Case studies/examples searches
        - Comparative/alternative perspective searches

        Create searches that complement each other and would provide a researcher with comprehensive information to write a detailed report."""
    return INSTRUCTIONS



class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")

def generate_planner_agent(n_searches: int):
    INSTRUCTIONS = generate_planner_agent_instructions(n_searches)
    planner_agent = Agent(
        name="PlannerAgent",
        instructions=INSTRUCTIONS,
        model="gpt-4o-mini",
        output_type=WebSearchPlan,
    )
    return planner_agent