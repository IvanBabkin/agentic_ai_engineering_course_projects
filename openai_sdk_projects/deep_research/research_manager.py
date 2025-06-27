from agents import Runner, trace, gen_trace_id
from search_agent import search_agent
from planner_agent import generate_planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from clarification_agent import clarification_agent, ClarificationPlan
import asyncio

# Import constants from main module
MIN_SEARCHES = 1
MAX_SEARCHES = 5
DEFAULT_SEARCHES = 3

def validate_n_searches(n_searches):
    """Validate and normalize the number of searches"""
    try:
        n_searches = int(n_searches) if n_searches else DEFAULT_SEARCHES
        # Apply guardrails
        if n_searches < MIN_SEARCHES:
            n_searches = MIN_SEARCHES
        elif n_searches > MAX_SEARCHES:
            n_searches = MAX_SEARCHES
        return n_searches
    except (ValueError, TypeError):
        return DEFAULT_SEARCHES

class ResearchManager:

    async def run(self, query: str, n_searches: int, clarification_answers: dict = None, trace_id: str = None):
        """ Run the deep research process, yielding the status updates and the final report"""
        # Validate n_searches at the start
        n_searches = validate_n_searches(n_searches)
        
        # Use provided trace_id or create a new one
        if trace_id is None:
            trace_id = gen_trace_id()
        
        # Always yield trace link for new sessions, but not when continuing from clarification
        if clarification_answers is None:
            yield f"🔍 [View trace](https://platform.openai.com/traces/trace?trace_id={trace_id})"
        
        # Use trace context manager properly within this function
        with trace("Deep Research Session", trace_id=trace_id):
            # Clarification step - Only run if clarification_answers is None (first attempt)
            if clarification_answers is None:
                yield "## Analyzing query complexity..."
                
                try:
                    result = await Runner.run(
                        clarification_agent,
                        f"Query: {query}",
                    )
                    
                    clarification_plan = result.final_output_as(ClarificationPlan)
                    
                    # Check if we should ask questions
                    if clarification_plan.should_ask_questions:
                        # Store reasoning for later use in progress log
                        self._clarification_reasoning = clarification_plan.assessment.reasoning
                        self._clarification_complexity = clarification_plan.assessment.complexity
                        
                        clarification_json = clarification_plan.model_dump_json()
                        # Include trace_id in the clarification data
                        yield f"CLARIFICATION_NEEDED:{clarification_json}|TRACE_ID:{trace_id}"
                        return
                    else:
                        pass
                        
                except Exception as e:
                    # Continue without clarification if there's an error
                    pass
            else:
                # Log clarification results to progress history
                yield "## Clarification Questions & Answers"
                
                # Show why clarification was asked (if we have the info from the plan)
                if hasattr(self, '_clarification_reasoning'):
                    yield f"**Why clarification was requested:** {self._clarification_reasoning} \n"
                    yield f"**Query complexity level:** {self._clarification_complexity}/3 \n"
                    yield ""
                
                if clarification_answers != "SKIP":
                    for i, (question, answer) in enumerate(clarification_answers.items(), 1):
                        yield f"**Question {i}:** {question} \t"
                        if answer == "Answer skipped":
                            yield f"**Answer:** *[Skipped]* \t"
                        elif not answer or answer.strip() == "":
                            yield f"**Answer:** *[Left blank]* \t"
                        else:
                            yield f"**Answer:** {answer} \t"
                        yield ""  # Add spacing between Q&A pairs
                else:
                    yield "*All clarification questions were skipped*"
                yield ""  # Add spacing after clarification section
            
            # Enhanced context for planner - handle SKIP case
            if clarification_answers == "SKIP":
                clarification_answers = None  # Reset to None for enhanced query building
            
            enhanced_query = self.build_enhanced_query(query, clarification_answers)
            
            # Plan searches (now with enhanced context)
            yield "## Generating search plan..."
            
            planner_agent = generate_planner_agent(n_searches)
            result = await Runner.run(
                planner_agent,
                enhanced_query,
            )
            search_plan = result.final_output_as(WebSearchPlan)
            
            # Display deviation reasoning if provided
            if search_plan.deviation_reasoning:
                if len(search_plan.searches) != n_searches:
                    yield f"**Search count adjusted:** {len(search_plan.searches)} searches (you requested {n_searches}) \n"
                    yield f"**Reasoning:** {search_plan.deviation_reasoning} \n"
                else:
                    yield f"**Note:** {search_plan.deviation_reasoning} \n"
            
            # Ensure we don't exceed the validated number of searches (this is now mostly redundant due to Pydantic constraints)
            if len(search_plan.searches) > MAX_SEARCHES:
                search_plan.searches = search_plan.searches[:MAX_SEARCHES]
            
            yield f"Will perform **{len(search_plan.searches)}** searches \n"
            for search in search_plan.searches:
                yield f"Query: **{search.query}** \t"
                yield f"Reason: {search.reason} \n"
            
            # Perform searches
            yield "## Searching..."
            num_completed = 0
            tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
            results = []
            for task in asyncio.as_completed(tasks):
                result = await task
                if result is not None:
                    results.append(result)
                num_completed += 1
                yield f"Searching... {num_completed}/{len(tasks)} completed \n"
            yield "Finished searching \n"
          
            # Write report
            yield "## Thinking about report..."
            input = f"Original query: {query}\n Summarized search results: {results}"
            result = await Runner.run(
                writer_agent,
                input,
            )
            report = result.final_output_as(ReportData)
            yield "Finished writing report"
            
            yield report.markdown_report

    async def search(self, item: WebSearchItem) -> str | None:
        """ Perform a search for the query """
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(
                search_agent,
                input,
            )
            return str(result.final_output)
        # return "test"
        except Exception:
            return None

    def build_enhanced_query(self, original_query: str, clarification_answers: dict = None) -> str:
        """Build enhanced query context from original query and clarification answers"""
        if not clarification_answers:
            return f"Query: {original_query}"
        
        enhanced = f"Original Query: {original_query}\n\nAdditional Context from User:\n"
        
        for question, answer in clarification_answers.items():
            enhanced += f"Q: {question}\nA: {answer}\n\n"
        
        return enhanced
