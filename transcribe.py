import gradio as gr
from split.state import get_split_summary

def create_transcribe_tab():
    """Create and configure the transcribe tab UI components.
    
    Returns:
        tuple: A tuple containing (tab_item, components)
    """
    with gr.TabItem("Transcribe") as transcribe_tab:
        with gr.Row():
            with gr.Column(scale=2):
                summary_display = gr.Textbox(
                    label="Audio Chunks Summary",
                    value="Loading...",
                    lines=10,
                    interactive=False
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh Summary")
        
        # Wire up event handlers
        transcribe_tab.select(fn=update_summary, inputs=[], outputs=[summary_display])
        refresh_btn.click(fn=update_summary, inputs=[], outputs=[summary_display])
        
        return transcribe_tab, summary_display

def update_summary():
    """Update the summary display with current split state information.
    
    Returns:
        str: Formatted summary of the current split state
    """
    return get_split_summary()