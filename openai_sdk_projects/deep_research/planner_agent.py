from pydantic import BaseModel, Field
from agents import Agent

# Import constants to ensure consistency
MIN_SEARCHES = 1
MAX_SEARCHES = 5

def generate_planner_agent_instructions(n_searches: int):
    INSTRUCTIONS = f"""
        You are an expert research strategist. Given a query (possibly with additional clarification context), create a comprehensive search plan.

        **USER PREFERENCE:** The user suggested {n_searches} searches, but you should determine the OPTIMAL number based on the query complexity.
        **GUARDRAILS:** You must generate between {MIN_SEARCHES} and {MAX_SEARCHES} searches (inclusive).

        **YOUR PRIMARY GOAL:** Determine the ideal number of searches for this specific query:
        
        **SIMPLE QUERIES (1-2 searches optimal):**
        - Basic definitions (e.g., "What is photosynthesis?")
        - Simple factual questions (e.g., "What is the capital of France?")
        - Single-concept explanations (e.g., "How does a microwave work?")
        
        **MODERATE QUERIES (2-3 searches optimal):**
        - Technology overviews (e.g., "What is blockchain?")
        - Process explanations (e.g., "How does machine learning work?")
        - Comparing 2-3 concepts
        
        **COMPLEX QUERIES (3-5 searches optimal):**
        - Multi-faceted topics (e.g., "AI impact on healthcare")
        - Broad subjects requiring multiple perspectives
        - Topics needing historical + current + future views
        - Industry analysis with multiple stakeholders

        **CRITICAL:** Always explain your reasoning in the deviation_reasoning field, even if you use the suggested number. Explain why that number is optimal for this query.

        When additional context is provided from clarification questions, use that information to:
        - Focus searches on the specific aspects the user is most interested in
        - Adjust the scope and depth based on their preferences
        - Target the appropriate audience level (technical vs. general)
        - Consider the specified time frame or geographical scope
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

        Create searches that complement each other and would provide a researcher with comprehensive information to write a detailed report.
        
        **EFFICIENCY PRINCIPLE:** Use the minimum number of searches needed to thoroughly answer the query. Don't create redundant searches just to reach a target number."""
    return INSTRUCTIONS


class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(
        description="A list of web searches to perform to best answer the query.",
        min_length=MIN_SEARCHES,
        max_length=MAX_SEARCHES
    )
    deviation_reasoning: str = Field(
        description="REQUIRED: Explain why you chose this specific number of searches and how it's optimal for this query complexity.",
        default=""
    )

def generate_planner_agent(n_searches: int):
    INSTRUCTIONS = generate_planner_agent_instructions(n_searches)
    planner_agent = Agent(
        name="PlannerAgent",
        instructions=INSTRUCTIONS,
        model="gpt-4o-mini",
        output_type=WebSearchPlan,
    )
    return planner_agent