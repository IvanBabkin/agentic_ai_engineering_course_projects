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
                # yield "**DEBUG:** Entering clarification step..."
                yield "## Analyzing query complexity..."
                
                try:
                    # yield f"**DEBUG:** About to call clarification agent with query: '{query}'"
                    
                    result = await Runner.run(
                        clarification_agent,
                        f"Query: {query}",
                    )
                    
                    # yield f"**DEBUG:** Clarification agent completed successfully"
                    
                    clarification_plan = result.final_output_as(ClarificationPlan)
                    
                    # yield f"**DEBUG:** Complexity: {clarification_plan.assessment.complexity}"
                    # yield f"**DEBUG:** Should ask questions: {clarification_plan.should_ask_questions}"
                    # yield f"**DEBUG:** Number of questions: {len(clarification_plan.questions)}"
                    # yield f"**DEBUG:** Reasoning: {clarification_plan.assessment.reasoning}"
                    
                    # Print the actual questions
                    # for i, q in enumerate(clarification_plan.questions):
                    #     yield f"**DEBUG:** Question {i+1}: {q.question}"
                    #     yield f"**DEBUG:** Purpose {i+1}: {q.purpose}"
                    
                    # Check if we should ask questions
                    if clarification_plan.should_ask_questions:
                        # Store reasoning for later use in progress log
                        self._clarification_reasoning = clarification_plan.assessment.reasoning
                        self._clarification_complexity = clarification_plan.assessment.complexity
                        
                        # yield f"**DEBUG:** About to yield CLARIFICATION_NEEDED with data"
                        clarification_json = clarification_plan.model_dump_json()
                        # yield f"**DEBUG:** JSON data: {clarification_json}"
                        # Include trace_id in the clarification data
                        yield f"CLARIFICATION_NEEDED:{clarification_json}|TRACE_ID:{trace_id}"
                        # yield f"**DEBUG:** Returned after yielding CLARIFICATION_NEEDED"
                        return
                    else:
                        # yield "**DEBUG:** No clarification needed, proceeding with search..."
                        pass
                        
                except Exception as e:
                    # yield f"**ERROR:** Exception in clarification step: {str(e)}"
                    # import traceback
                    # yield f"**ERROR:** Full traceback: {traceback.format_exc()}"
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
            
            # DEBUG: Context analysis before planning
            # yield f"**DEBUG CONTEXT:** Original query: '{query}'"
            # yield f"**DEBUG CONTEXT:** Clarification answers after processing: {clarification_answers}"
            # yield f"**DEBUG CONTEXT:** Has clarification context: {clarification_answers is not None and len(clarification_answers) > 0 if clarification_answers else False}"
            
            # if clarification_answers:
            #     yield f"**DEBUG CONTEXT:** Number of clarification Q&As: {len(clarification_answers)}"
            #     for i, (question, answer) in enumerate(clarification_answers.items(), 1):
            #         yield f"**DEBUG CONTEXT:** Q{i}: {question[:50]}..."
            #         yield f"**DEBUG CONTEXT:** A{i}: {answer}"
            #         if answer == "Answer skipped":
            #             yield f"**DEBUG CONTEXT:** ↳ This question was skipped by user"
            #         else:
            #             yield f"**DEBUG CONTEXT:** ↳ This provides specific user context"
            
            # DEBUG: Enhanced query building
            # yield f"**DEBUG ENHANCE:** Building enhanced query..."
            # yield f"**DEBUG ENHANCE:** Original query: '{query}'"
            
            # if not clarification_answers:
            #     yield f"**DEBUG ENHANCE:** No clarification answers - returning basic query"
            # else:
            #     yield f"**DEBUG ENHANCE:** Processing {len(clarification_answers)} clarification answers"
            #     for question, answer in clarification_answers.items():
            #         yield f"**DEBUG ENHANCE:** Added Q&A pair - Answer type: {'Skipped' if answer == 'Answer skipped' else 'Provided'}"
            
            enhanced_query = self.build_enhanced_query(query, clarification_answers)
            
            # yield f"**DEBUG ENHANCE:** Enhanced query total length: {len(enhanced_query)} characters"
            # if clarification_answers:
            #     yield f"**DEBUG ENHANCE:** Enhancement includes {'skipped and answered' if any(a != 'Answer skipped' for a in clarification_answers.values()) and any(a == 'Answer skipped' for a in clarification_answers.values()) else 'all answered' if all(a != 'Answer skipped' for a in clarification_answers.values()) else 'all skipped'} questions"
            
            # yield f"**DEBUG PLANNING:** Enhanced query length: {len(enhanced_query)} characters"
            # yield f"**DEBUG PLANNING:** Enhanced query preview: {enhanced_query[:200]}..."
            
            # Plan searches (now with enhanced context)
            yield "## Generating search plan..."
            # yield f"**DEBUG PLANNING:** About to call planner with enhanced context"
            # yield f"**DEBUG PLANNING:** Context enhancement factor: {'High' if clarification_answers and len(clarification_answers) > 2 else 'Medium' if clarification_answers else 'None'}"
            
            planner_agent = generate_planner_agent(n_searches)
            result = await Runner.run(
                planner_agent,
                enhanced_query,
            )
            search_plan = result.final_output_as(WebSearchPlan)
            
            # Ensure we don't exceed the validated number of searches
            if len(search_plan.searches) > n_searches:
                search_plan.searches = search_plan.searches[:n_searches]
            
            yield f"Will perform **{len(search_plan.searches)}** searches (validated: {n_searches} requested) \n"
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
            # result = await Runner.run(
            #     search_agent,
            #     input,
            # )
            # return str(result.final_output)
            return "test"
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
