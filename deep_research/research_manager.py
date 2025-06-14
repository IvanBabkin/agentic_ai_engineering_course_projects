from agents import Runner, trace, gen_trace_id
from search_agent import search_agent
from planner_agent import generate_planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from clarification_agent import clarification_agent, ClarificationPlan
import asyncio

class ResearchManager:

    async def run(self, query: str, n_searches: int, clarification_answers: dict = None):
        """ Run the deep research process, yielding the status updates and the final report"""
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            trace_url = f"https://platform.openai.com/traces/trace?trace_id={trace_id}"
            yield f"🔍 [View trace]({trace_url})"
            
            # Plan searches
            yield "## Generating search plan..."
            planner_agent = generate_planner_agent(n_searches)
            result = await Runner.run(
                planner_agent,
                f"Query: {query}",
            )
            search_plan = result.final_output_as(WebSearchPlan)
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
