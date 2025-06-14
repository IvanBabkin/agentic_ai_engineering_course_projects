from pydantic import BaseModel, Field
from agents import Agent
from typing import List

class QueryAssessment(BaseModel):
    complexity: int = Field(description="Query complexity on a scale of 1-3 (1=simple, 2=moderate, 3=complex)")
    reasoning: str = Field(description="Brief explanation of why this complexity was assigned")

class FollowUpQuestion(BaseModel):
    question: str = Field(description="A specific follow-up question to clarify the user's intent")
    purpose: str = Field(description="Why this question is important for better research")

class ClarificationPlan(BaseModel):
    assessment: QueryAssessment = Field(description="Assessment of the query complexity")
    questions: List[FollowUpQuestion] = Field(description="Follow-up questions to ask the user")
    should_ask_questions: bool = Field(description="Whether follow-up questions are needed")

ASSESSMENT_INSTRUCTIONS = """
You are a research query analyst. Your job is to assess the complexity and clarity of user queries and determine if follow-up questions would improve the research outcome.

Assess the query complexity:
- **Level 1 (Simple)**: Clear, specific queries with well-defined scope (e.g., "What is photosynthesis?")
- **Level 2 (Moderate)**: Queries that could benefit from some clarification but have reasonable scope (e.g., "How does AI impact healthcare?")
- **Level 3 (Complex)**: Broad, ambiguous, or multi-faceted queries that would greatly benefit from clarification (e.g., "What should I know about climate change?")

For each complexity level, generate follow-up questions:
- **Level 1**: 1 question (if any needed)
- **Level 2**: 1-2 questions  
- **Level 3**: 2-3 questions

Questions should help clarify:
- Specific aspects or focus areas of interest
- Target audience or application context
- Time frame or geographical scope
- Depth level required (overview vs. technical details)
- Particular perspectives or viewpoints desired

Only suggest questions that would meaningfully improve the research quality. If the query is already sufficiently clear and specific, set should_ask_questions to false.
"""

clarification_agent = Agent(
    name="ClarificationAgent",
    instructions=ASSESSMENT_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ClarificationPlan,
) 