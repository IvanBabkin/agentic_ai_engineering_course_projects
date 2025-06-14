import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager

load_dotenv(override=True)

async def run(query: str, history: str, n_searches: int):
    # Reset history when a new query starts
    history = "# Deep Research Progress\n"

    async for chunk in ResearchManager().run(query, n_searches):
        # Check if this is the final report (it will be the last chunk)
        if chunk.startswith("# Final report for"):
            # For the final report, update both report and history
            final_report = "--- \n" + chunk
            yield history, final_report, history
        else:
            # For status updates, add to history and format as markdown
            history = history + '\n' + chunk if history else chunk
            yield history, "", history
    

with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# <div style='text-align: center; font-size: 2.5em; margin: 20px 0;'>Deep Research Light</div>")
    query_textbox = gr.Textbox(label="What would you like to research?")
    n_searches_textbox = gr.Textbox(label="How many searches would you like to perform?")
    run_button = gr.Button("Run", variant="primary")
    
    # Status updates as a clean markdown section
    status = gr.Markdown(
        value="",
        elem_classes=["status-updates"]  # Add custom class for styling if needed
    )
    
    
    # Final report in markdown
    report = gr.Markdown(
        value=""
    )
    
    history = gr.State("")  # Initialize state with empty string
    
    run_button.click(
        fn=run, 
        inputs=[query_textbox, history, n_searches_textbox], 
        outputs=[status, report, history],
        api_name="run",
        queue=True
    )
    query_textbox.submit(
        fn=run, 
        inputs=[query_textbox, history, n_searches_textbox], 
        outputs=[status, report, history],
        api_name="run",
        queue=True
    )

ui.launch(inbrowser=True)

