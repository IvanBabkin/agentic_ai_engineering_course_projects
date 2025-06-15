import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager
import json

load_dotenv(override=True)

# Global constants for search limits
MIN_SEARCHES = 1
MAX_SEARCHES = 5
DEFAULT_SEARCHES = 3

def validate_n_searches(n_searches_input):
    """Validate and normalize the number of searches input"""
    try:
        n_searches = int(n_searches_input) if n_searches_input else DEFAULT_SEARCHES
        # Apply guardrails
        if n_searches < MIN_SEARCHES:
            n_searches = MIN_SEARCHES
        elif n_searches > MAX_SEARCHES:
            n_searches = MAX_SEARCHES
        return n_searches
    except (ValueError, TypeError):
        return DEFAULT_SEARCHES

async def run_research(query: str, n_searches: int, clarification_answers_json: str = None, trace_id: str = None, assessment: dict = None):
    """Run research with optional clarification answers and trace_id"""
    # Parse clarification answers - distinguish between first attempt and skip/empty
    try:
        if clarification_answers_json is None:
            # First attempt - no clarification attempted yet
            clarification_answers = None
        elif clarification_answers_json == "{}":
            # Skip clarification or empty answers - proceed with research
            clarification_answers = "SKIP"
        else:
            # Parse actual answers
            clarification_answers = json.loads(clarification_answers_json)
    except:
        clarification_answers = None
    
    # Initialize or continue history - preserve trace link when continuing
    if trace_id and clarification_answers is not None:
        # Continuing from clarification - preserve trace link
        history = f"# Deep Research Progress\n\n🔍 [View trace](https://platform.openai.com/traces/trace?trace_id={trace_id})\n"
    else:
        # Starting fresh
        history = "# Deep Research Progress\n"

    # Create a new research manager for each request to avoid context issues
    research_manager = ResearchManager()
    
    # Pass assessment info to research manager if available
    if assessment:
        research_manager._clarification_reasoning = assessment.get('reasoning', '')
        research_manager._clarification_complexity = assessment.get('complexity', 1)
    
    async for chunk in research_manager.run(query, n_searches, clarification_answers, trace_id):
        # Check if clarification is needed
        if chunk.startswith("CLARIFICATION_NEEDED:"):
            # Parse the clarification data and trace_id
            if "|TRACE_ID:" in chunk:
                clarification_part, trace_part = chunk.split("|TRACE_ID:")
                clarification_data = clarification_part.replace("CLARIFICATION_NEEDED:", "")
                extracted_trace_id = trace_part
            else:
                clarification_data = chunk.replace("CLARIFICATION_NEEDED:", "")
                extracted_trace_id = None
            
            # IMPORTANT: Pass the raw clarification data through a hidden channel
            # Don't show it in status or report - only in history for processing
            yield history, "", f"CLARIFICATION_NEEDED:{clarification_data}|TRACE_ID:{extracted_trace_id}"
            return
        
        # Check if this is the final report (it will be the last chunk)
        if chunk.startswith("# Final report for"):
            # For the final report, update both report and history
            final_report = "--- \n" + chunk
            yield history, final_report, history
        else:
            # For status updates, add to history and format as markdown
            if chunk.strip():  # Only add non-empty chunks
                history = history + '\n' + chunk if history.strip() else chunk
            yield history, "", history

def parse_clarification_questions(clarification_data):
    """Parse clarification data and return individual questions"""
    try:
        data = json.loads(clarification_data)
        questions = data.get('questions', [])
        return questions
    except Exception as e:
        return []

def create_clarification_interface(questions, assessment=None):
    """Dynamically create clarification interface based on number of questions"""
    interface_html = "### 🤔 Please answer these questions to improve your research:\n\n"
    
    # Add reasoning if available
    if assessment:
        complexity_levels = {1: "Simple", 2: "Moderate", 3: "Complex"}
        complexity_desc = complexity_levels.get(assessment.get('complexity', 1), "Unknown")
        interface_html += f"**Query Complexity:** {complexity_desc} (Level {assessment.get('complexity', 1)}/3)\n\n"
        interface_html += f"**Why we're asking:** {assessment.get('reasoning', 'To better understand your research needs')}\n\n"
        interface_html += "---\n\n"
    
    for i, question in enumerate(questions, 1):
        interface_html += f"**Question {i}:** {question['question']}\n\n"
        interface_html += f"*Purpose: {question['purpose']}*\n\n"
        interface_html += f"<div style='margin-bottom: 20px;'></div>\n\n"
    
    return interface_html

def handle_research_output(status_output, report_output, history_output):
    """Handle research output and check for clarification needs"""
    # Check history_output instead of report_output for clarification signals
    if isinstance(history_output, str) and history_output.startswith("CLARIFICATION_NEEDED:"):
        # Parse the clarification data and trace_id
        if "|TRACE_ID:" in history_output:
            clarification_part, trace_part = history_output.split("|TRACE_ID:")
            clarification_data = clarification_part.replace("CLARIFICATION_NEEDED:", "")
            trace_id = trace_part
        else:
            clarification_data = history_output.replace("CLARIFICATION_NEEDED:", "")
            trace_id = None
        
        # Parse the full clarification plan to get assessment info
        try:
            clarification_plan = json.loads(clarification_data)
            questions = clarification_plan.get('questions', [])
            assessment = clarification_plan.get('assessment', {})
        except:
            questions = parse_clarification_questions(clarification_data)
            assessment = None
        
        # Create the clarification interface with assessment info
        clarification_html = create_clarification_interface(questions, assessment)
        
        # Store assessment info for later use in progress log
        assessment_json = json.dumps(assessment) if assessment else ""
        
        # Return proper history without the CLARIFICATION_NEEDED signal
        clean_history = status_output  # Use status as the clean history
        
        return (
            clean_history,  # status shows clean progress
            clarification_html,  # show clarification questions with reasoning
            clean_history,  # history shows clean progress
            gr.update(visible=True),  # show clarification section
            json.dumps(questions),  # store questions as JSON
            True,  # clarification needed flag
            gr.update(visible=True, value="", lines=max(2, len(questions))),  # show answer textbox with appropriate height
            trace_id,  # store trace_id
            assessment_json  # store assessment info
        )
    else:
        return (
            status_output,
            report_output,
            history_output,
            gr.update(visible=False),  # hide clarification section
            "",  # clear stored questions
            False,  # no clarification needed
            gr.update(visible=False, value=""),  # hide answer textbox
            "",  # clear trace_id
            ""  # clear assessment info
        )

async def handle_clarification_submit(query, n_searches_input, stored_questions_json, answers_text, clarification_needed, trace_id, assessment_json):
    """Handle clarification submission with flexible answer parsing"""
    if not clarification_needed:
        yield gr.update(), gr.update(), gr.update(), gr.update(visible=False), "", False, gr.update(visible=False), "", ""
        return
    
    n_searches = validate_n_searches(n_searches_input)
    
    try:
        # Parse stored questions and assessment
        questions = json.loads(stored_questions_json) if stored_questions_json else []
        assessment = json.loads(assessment_json) if assessment_json else {}
        
        # Parse answers - support multiple formats and preserve blanks
        answers = []
        if answers_text:
            # Split by newlines and preserve empty lines as blank answers
            raw_answers = answers_text.split('\n')
            answers = [a.strip() for a in raw_answers]
        
        # Create responses dictionary - map questions to answers by index
        responses_dict = {}
        for i, question in enumerate(questions):
            if i < len(answers):
                answer = answers[i] if answers[i] else ""  # Preserve empty answers
                responses_dict[question['question']] = answer if answer else "Answer skipped"
            else:
                responses_dict[question['question']] = "Answer skipped"
        
        # Continue research with clarification answers, assessment, and same trace_id
        async for result in run_research(query, n_searches, json.dumps(responses_dict), trace_id, assessment):
            # Unpack the result tuple and add the state values
            status, report, history = result
            yield status, report, history, gr.update(visible=False), "", False, gr.update(visible=False), "", ""
        
    except Exception as e:
        error_msg = f"Error processing clarification responses: {str(e)}"
        yield error_msg, "", "", gr.update(visible=False), "", False, gr.update(visible=False), "", ""

async def start_research(query, n_searches_input):
    """Start the research process"""
    n_searches = validate_n_searches(n_searches_input)
    
    if not query.strip():
        yield "Please enter a research question.", "", ""
        return
    
    async for result in run_research(query, n_searches):
        yield result

async def skip_clarification(query, n_searches_input, trace_id, stored_questions_json, assessment_json):
    """Skip clarification and proceed with original query in same trace, showing skipped answers"""
    n_searches = validate_n_searches(n_searches_input)
    
    # Parse stored questions and assessment
    try:
        questions = json.loads(stored_questions_json) if stored_questions_json else []
        assessment = json.loads(assessment_json) if assessment_json else {}
        
        # Create responses dictionary with "Answer skipped" for each question
        responses_dict = {}
        for question in questions:
            responses_dict[question['question']] = "Answer skipped"
        
        # Continue with skipped answers, assessment, and same trace_id
        async for result in run_research(query, n_searches, json.dumps(responses_dict), trace_id, assessment):
            # Unpack the result tuple and add the state values for consistency
            status, report, history = result
            yield status, report, history, gr.update(visible=False), "", False, gr.update(visible=False), "", ""
    
    except Exception as e:
        error_msg = f"Error processing skip clarification: {str(e)}"
        yield error_msg, "", "", gr.update(visible=False), "", False, gr.update(visible=False), "", ""

def show_search_validation(n_searches_input):
    """Show validation feedback for number of searches"""
    try:
        n_searches = int(n_searches_input) if n_searches_input else DEFAULT_SEARCHES
        if n_searches < MIN_SEARCHES:
            return f"⚠️ Minimum {MIN_SEARCHES} search required. Using {MIN_SEARCHES}."
        elif n_searches > MAX_SEARCHES:
            return f"⚠️ Maximum {MAX_SEARCHES} searches allowed. Using {MAX_SEARCHES}."
        else:
            return f"✅ Will perform {n_searches} searches."
    except (ValueError, TypeError):
        return f"⚠️ Invalid input. Using default: {DEFAULT_SEARCHES} searches."

with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# <div style='text-align: center; font-size: 2.5em; margin: 20px 0;'>Deep Research Light</div>")
    
    # Input section
    with gr.Row():
        query_textbox = gr.Textbox(
            label="What would you like to research?", 
            placeholder="Enter your research question...",
            scale=3
        )
        n_searches_textbox = gr.Textbox(
            label=f"Number of searches", 
            placeholder=f"Enter number of searches (between {MIN_SEARCHES} and {MAX_SEARCHES})",
            scale=1,
        )
    
    # Search validation feedback
    search_validation = gr.Markdown(value="", visible=False)
    
    run_button = gr.Button("Start Research", variant="primary", size="lg")
    
    # Status updates
    status = gr.Markdown(value="", label="Research Progress")
    
    # Final report
    report = gr.Markdown(value="", label="Research Report")
    
    # Clarification interface (initially hidden)
    with gr.Column(visible=False) as clarification_section:
        clarification_display = gr.Markdown()
        
        clarification_answers = gr.Textbox(
            label="Your Answers (one per line, in order)",
            placeholder="Answer to question 1\nAnswer to question 2\nAnswer to question 3",
            lines=3,
            visible=False
        )
        
        gr.Markdown("**Instructions:** Please provide your answers in order, one per line. You can skip questions by leaving blank lines.")
        
        with gr.Row():
            submit_clarification = gr.Button("Continue with Research", variant="primary")
            cancel_clarification = gr.Button("Skip Clarification", variant="secondary")
    
    # Hidden state variables
    history = gr.State("")
    stored_questions = gr.State("")
    clarification_needed = gr.State(False)
    current_trace_id = gr.State("")  # Store trace_id across interactions
    stored_assessment = gr.State("")  # Store assessment info across interactions
    
    # Main research button
    research_output = run_button.click(
        fn=start_research,
        inputs=[query_textbox, n_searches_textbox],
        outputs=[status, report, history],
        api_name="run",
        queue=True
    )
    
    # Handle clarification detection
    research_output.then(
        fn=handle_research_output,
        inputs=[status, report, history],
        outputs=[status, clarification_display, history, clarification_section, stored_questions, clarification_needed, clarification_answers, current_trace_id, stored_assessment]
    )
    
    # Handle clarification submission
    submit_clarification.click(
        fn=handle_clarification_submit,
        inputs=[query_textbox, n_searches_textbox, stored_questions, clarification_answers, clarification_needed, current_trace_id, stored_assessment],
        outputs=[status, report, history, clarification_section, stored_questions, clarification_needed, clarification_answers, current_trace_id, stored_assessment],
        queue=True
    )
    
    # Handle skip clarification
    cancel_clarification.click(
        fn=skip_clarification,
        inputs=[query_textbox, n_searches_textbox, current_trace_id, stored_questions, stored_assessment],
        outputs=[status, report, history, clarification_section, stored_questions, clarification_needed, clarification_answers, current_trace_id, stored_assessment],
        queue=True
    )
    
    # Allow enter key submission
    query_textbox.submit(
        fn=start_research,
        inputs=[query_textbox, n_searches_textbox],
        outputs=[status, report, history],
        queue=True
    ).then(
        fn=handle_research_output,
        inputs=[status, report, history],
        outputs=[status, clarification_display, history, clarification_section, stored_questions, clarification_needed, clarification_answers, current_trace_id]
    )

    # Show validation when user changes the number of searches
    n_searches_textbox.change(
        fn=show_search_validation,
        inputs=[n_searches_textbox],
        outputs=[search_validation]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[search_validation]
    )

if __name__ == "__main__":
    ui.launch(inbrowser=True)
