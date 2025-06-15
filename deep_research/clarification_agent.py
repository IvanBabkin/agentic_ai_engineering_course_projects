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

IMPORTANT: You should almost always ask clarification questions to improve research quality. Only skip questions for extremely simple, well-defined queries.

Assess the query complexity:
- **Level 1 (Simple)**: Clear, specific queries with well-defined scope (e.g., "What is photosynthesis?")
- **Level 2 (Moderate)**: Queries that could benefit from clarification (e.g., "How does AI impact healthcare?")
- **Level 3 (Complex)**: Broad, ambiguous, or multi-faceted queries (e.g., "What should I know about climate change?")

For each complexity level, you MUST generate follow-up questions:
- **Level 1**: Ask 1 question to narrow scope or identify specific aspects
- **Level 2**: Ask 2 questions to clarify context and focus
- **Level 3**: Ask 3 questions to define scope, audience, and specific interests

Questions should help clarify:
- Specific aspects or focus areas of interest
- Target audience or application context
- Time frame or geographical scope
- Depth level required (overview vs. technical details)
- Particular perspectives or viewpoints desired

CRITICAL: Set should_ask_questions to True unless the query is extremely simple and specific (like "Define photosynthesis"). For 95% of queries, you should ask clarifying questions.

Examples of when to ask questions:
- "AI in healthcare" → Ask about specific applications, timeline, audience
- "Climate change" → Ask about specific aspects, geographic focus, perspective
- "Best programming language" → Ask about use case, experience level, project type
- "Marketing strategies" → Ask about industry, budget, target audience

Only set should_ask_questions to False for queries like:
- "What is the capital of France?"
- "Define machine learning"
- "How many chromosomes do humans have?"
"""

clarification_agent = Agent(
    name="ClarificationAgent",
    instructions=ASSESSMENT_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ClarificationPlan,
) 