import os
import gradio as gr
import threading
import json
import shutil
import re
from datetime import datetime
from pydub import AudioSegment
from split.state import get_split_summary, get_split_state
from utils.logger import logger
from .transcriber import TranscriptionService
import time

LANGUAGE_MAP = {
    "Korean": "ko",
    "English": "en",
    "Chinese": "zh",
    "Vietnamese": "vi",
    "Spanish": "es"
}

# Ensure audio_text directory exists
AUDIO_TEXT_DIR = "audio_text"
os.makedirs(AUDIO_TEXT_DIR, exist_ok=True)

# Global variables
transcription_service = None
is_transcribing = False
transcription_thread = None
chunk_prompts = {}  # Store prompts used for each chunk

# Helper function to estimate tokens in text
def estimate_tokens(text):
    """Estimate the number of tokens in a text.
    
    This is a rough estimate based on the rule of thumb that 1 token ≈ 4 characters in English.
    For other languages, this might vary, but it's a reasonable approximation.
    
    Args:
        text (str): The text to estimate tokens for
        
    Returns:
        int: Estimated number of tokens
    """
    # Count characters (excluding whitespace)
    char_count = sum(1 for c in text if not c.isspace())
    # Estimate tokens (roughly 4 chars per token)
    return max(1, char_count // 4)

# Helper function to estimate speaking time in seconds
def estimate_speaking_time(text):
    """Estimate how long it would take to speak the text in seconds.
    
    Based on average speaking rate of ~150 words per minute or 2.5 words per second.
    
    Args:
        text (str): The text to estimate speaking time for
        
    Returns:
        float: Estimated speaking time in seconds
    """
    # Count words
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    # Estimate time (2.5 words per second is average speaking rate)
    return word_count / 2.5

# Helper function to trim text to fit within token limit and speaking time
def trim_text_for_prompt(text, max_tokens=224, max_speaking_time=None, min_tokens=40):
    """Trim text to fit within token limit and speaking time while preserving line breaks and full words.
    
    Args:
        text (str): The text to trim
        max_tokens (int): Maximum number of tokens allowed
        max_speaking_time (float, optional): Maximum speaking time in seconds
        min_tokens (int): Minimum number of tokens to preserve (to ensure adequate context)
        
    Returns:
        str: Trimmed text
    """
    if not text:
        return ""
    
    # If text is already within limits, return as is
    estimated_tokens = estimate_tokens(text)
    estimated_time = estimate_speaking_time(text)
    
    if estimated_tokens <= max_tokens and (max_speaking_time is None or estimated_time <= max_speaking_time):
        return text
    
    # Calculate target token count based on speaking time
    # For a 15-second overlap, we want approximately 60 tokens (15 sec * 2.5 words/sec * 1.6 tokens/word)
    target_tokens = max_tokens
    if max_speaking_time is not None:
        # Calculate tokens needed for the speaking time (2.5 words per second, ~1.6 tokens per word)
        time_based_tokens = int(max_speaking_time * 2.5 * 1.6)
        # Use the smaller of max_tokens or time_based_tokens, but not less than min_tokens
        # For a 15-second overlap, this should be around 60 tokens
        target_tokens = min(max_tokens, time_based_tokens)
        
        # Ensure we have at least the minimum tokens
        target_tokens = max(target_tokens, min_tokens)
        
        # For very short overlaps, we still want a reasonable amount of context
        # At least 40% of the maximum tokens or the calculated time-based tokens, whichever is larger
        if target_tokens < (max_tokens * 0.4):
            target_tokens = max(target_tokens, int(max_tokens * 0.4))
    
    # Split text into lines
    lines = text.split('\n')
    
    # Process the text to fit within our target token count
    # We'll try to keep as much of the most recent text as possible
    result_lines = []
    current_tokens = 0
    
    # First, try to include whole lines from the end
    for line in reversed(lines):
        line_tokens = estimate_tokens(line)
        
        # If this line would put us over the target, we need to process it differently
        if current_tokens + line_tokens > target_tokens:
            # If we already have some content and this line would exceed our target by a lot,
            # skip it and use what we have
            if current_tokens > 0 and current_tokens >= min_tokens:
                break
                
            # Otherwise, try to include part of this line to reach our target
            words = line.split()
            partial_line = []
            partial_tokens = 0
            
            # Add words from the end until we reach the target
            for word in reversed(words):
                word_tokens = estimate_tokens(word)
                if current_tokens + partial_tokens + word_tokens <= target_tokens:
                    partial_line.insert(0, word)
                    partial_tokens += word_tokens
                else:
                    # If we already have some words and adding this one would exceed our target,
                    # stop here
                    if partial_line:
                        break
                    
                    # If we don't have any words yet, include this one even if it exceeds our target
                    # This ensures we include at least one word
                    partial_line.insert(0, word)
                    partial_tokens += word_tokens
                    break
                    
            if partial_line:
                result_lines.insert(0, ' '.join(partial_line))
                current_tokens += partial_tokens
            break
        
        # Add the full line
        result_lines.insert(0, line)
        current_tokens += line_tokens
        
        # If we've reached our target and have enough tokens, stop
        if current_tokens >= target_tokens and current_tokens >= min_tokens:
            break
    
    # If we didn't get enough tokens from the end, add more from the beginning
    if current_tokens < min_tokens:
        # Start with the lines we haven't included yet
        remaining_lines = lines[:len(lines)-len(result_lines)]
        
        # Add lines from the beginning until we reach the minimum
        for line in remaining_lines:
            line_tokens = estimate_tokens(line)
            
            # If adding this line would exceed the maximum, skip it
            if current_tokens + line_tokens > max_tokens:
                continue
                
            # Add the line
            result_lines.append(line)
            current_tokens += line_tokens
            
            # If we've reached the minimum, stop
            if current_tokens >= min_tokens:
                break
    
    # Join the lines back together
    return '\n'.join(result_lines)

# Helper function to trim text to a specific character limit
def trim_text_to_char_limit(text, char_limit):
    """Trim text to a specific character limit while preserving full words.
    
    Args:
        text (str): The text to trim
        char_limit (int): Maximum number of characters to keep
        
    Returns:
        str: Trimmed text
    """
    if not text:
        return ""
    
    # If text is already within limit, return as is
    if len(text) <= char_limit:
        return text
    
    # Split text into lines
    lines = text.split('\n')
    
    # Process the text to fit within our character limit
    # We'll try to keep as much of the most recent text as possible
    result_lines = []
    current_chars = 0
    
    # First, try to include whole lines from the end
    for line in reversed(lines):
        line_chars = len(line)
        
        # If this line would put us over the limit, we need to process it differently
        if current_chars + line_chars > char_limit:
            # If we already have some content, skip this line
            if current_chars > 0:
                break
                
            # Otherwise, try to include part of this line to reach our target
            words = line.split()
            partial_line = []
            partial_chars = 0
            
            # Add words from the end until we reach the limit
            for word in reversed(words):
                word_chars = len(word) + 1  # +1 for the space
                if current_chars + partial_chars + word_chars <= char_limit:
                    partial_line.insert(0, word)
                    partial_chars += word_chars
                else:
                    # If we already have some words, stop here
                    if partial_line:
                        break
                    
                    # If we don't have any words yet, include this one even if it exceeds our limit
                    partial_line.insert(0, word)
                    partial_chars += word_chars
                    break
                    
            if partial_line:
                result_lines.insert(0, ' '.join(partial_line))
                current_chars += partial_chars
            break
        
        # Add the full line
        result_lines.insert(0, line)
        current_chars += line_chars + 1  # +1 for the newline
        
        # If we've reached our limit, stop
        if current_chars >= char_limit:
            break
    
    # Join the lines back together
    return '\n'.join(result_lines)

def create_transcribe_tab():
    """Create the transcribe tab UI.
    
    Returns:
        tuple: The transcribe tab and summary display
    """
    with gr.TabItem("Transcribe") as transcribe_tab:
        with gr.Row():
            with gr.Column(scale=2):
                # Transcription settings
                gr.Markdown("### Transcription Settings")
                
                # Language selection
                language = gr.Dropdown(
                    choices=list(LANGUAGE_MAP.keys()),
                    value="English",
                    label="Language"
                )
                
                # Model path
                model_path = gr.Textbox(
                    value=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "whisper.cpp", "models", "ggml-large-v3-turbo.bin"),
                    label="Whisper.cpp Model Path",
                    info="Path to the whisper.cpp model file"
                )
                
                # Prompt settings
                gr.Markdown("### Prompt Settings")
                
                # Master toggle for using prompts at all
                use_prompt = gr.Checkbox(
                    label="Use Prompt",
                    value=True,
                    info="When disabled, no prompts will be used for any chunks"
                )
                
                with gr.Group(visible=True) as prompt_options:
                    # Radio buttons for prompt types
                    prompt_type = gr.Radio(
                        choices=["No Prompt", "Previous Chunk", "Custom Prompt"],
                        value="Previous Chunk",
                        label="Prompt Type",
                        info="Select the type of prompt to use for transcription"
                    )
                    
                    # Show prompts in transcription checkbox (only visible when Previous Chunk is selected)
                    show_prompts = gr.Checkbox(
                        label="Show prompt per chunk in Transcription",
                        value=True,
                        visible=True,
                        info="When enabled, shows the prompt used for each chunk in the transcription display"
                    )
                    
                    # Previous Chunk options (only visible when Previous Chunk is selected)
                    with gr.Group(visible=True) as previous_chunk_options:
                        prompt_trim_type = gr.Radio(
                            choices=["Calculated (Automatic)", "Custom Size"],
                            value="Calculated (Automatic)",
                            label="Prompt Trimming Method",
                            info="Choose how to trim the previous chunk for use as a prompt"
                        )
                        
                        custom_char_limit = gr.Number(
                            value=300,
                            label="Custom Character Limit",
                            info="Maximum number of characters to use from the previous chunk (will trim to nearest full word)",
                            visible=False,
                            minimum=50,
                            maximum=1000,
                            step=10
                        )
                    
                    # Custom prompt text field
                    custom_prompt = gr.Textbox(
                        label="Custom Prompt",
                        placeholder="Enter a custom prompt to use for all chunks...",
                        lines=2,
                        visible=False,
                        info="This prompt will be used for all chunks, including the first one.",
                        max_lines=5
                    )
                
                # Transcribe button
                with gr.Row():
                    transcribe_btn = gr.Button("Transcribe Audio Chunks")
                    stop_btn = gr.Button("Stop Transcription", interactive=False)
                
                # Status display
                status_display = gr.Markdown("Ready to transcribe")
            
                # Transcription output
                gr.Markdown("### Transcription Output")
                transcription_output = gr.Textbox(
                    label="Transcription",
                    placeholder="Transcription will appear here...",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )
            
            # Summary display
            with gr.Column(scale=1):
                gr.Markdown("### Audio Summary")
                summary_display = gr.Textbox(
                    label="Audio Chunks Summary",
                    value="Loading...",
                    lines=15,
                    interactive=False
                )
                refresh_btn = gr.Button("Refresh Summary")
                
                # Clear transcriptions button
                clear_transcriptions_btn = gr.Button("Clear All Transcriptions", variant="secondary")
                clear_status = gr.Markdown("")
        
        # Set up event handlers
        transcribe_tab.select(fn=update_summary, inputs=[], outputs=[summary_display])
        refresh_btn.click(fn=update_summary, inputs=[], outputs=[summary_display])
        
        # Update prompt options visibility based on use_prompt
        use_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_prompt],
            outputs=[prompt_options]
        )
        
        # Show/hide custom prompt field and previous chunk options based on prompt type
        prompt_type.change(
            fn=lambda x: (
                gr.update(visible=(x == "Custom Prompt")), 
                gr.update(visible=(x == "Previous Chunk")),
                gr.update(visible=(x == "Previous Chunk"))
            ),
            inputs=[prompt_type],
            outputs=[custom_prompt, show_prompts, previous_chunk_options]
        )
        
        # Show/hide custom character limit based on prompt trim type
        prompt_trim_type.change(
            fn=lambda x: gr.update(visible=(x == "Custom Size")),
            inputs=[prompt_trim_type],
            outputs=[custom_char_limit]
        )
        
        transcribe_btn.click(
            fn=start_transcription,
            inputs=[language, model_path, use_prompt, prompt_type, custom_prompt, show_prompts, prompt_trim_type, custom_char_limit],
            outputs=[status_display, transcribe_btn, stop_btn]
        )
        
        stop_btn.click(
            fn=stop_transcription,
            inputs=[],
            outputs=[status_display, transcribe_btn, stop_btn]
        )
        
        # Define a function to check transcription status and update UI accordingly
        def check_transcription_status():
            global is_transcribing
            content, status = update_transcription_display(show_prompts.value if hasattr(show_prompts, 'value') else False)
            
            # Check if transcription has completed by looking for completion message in content
            transcription_completed = "Transcription completed at" in content if content else False
            
            # If transcription was in progress but has now completed
            if not is_transcribing:
                # Update the status to show completion
                if "in progress" in status:
                    status = status.replace("in progress", "complete")
                elif "Transcription complete" not in status and "No transcription" not in status:
                    status = "Transcription complete. " + status
                
                # Return content, status, and button updates
                return content, status, gr.update(interactive=True), gr.update(interactive=False)
            
            # If transcription is still in progress
            if is_transcribing:
                return content, status, gr.update(interactive=False), gr.update(interactive=True)
            
            # If no transcription is in progress and we're just displaying a completed transcription
            return content, status, gr.update(interactive=True), gr.update(interactive=False)
        
        # Set up periodic refresh of transcription output (every 2 seconds)
        refresh_timer = gr.Timer(value=2)
        refresh_timer.tick(
            fn=check_transcription_status,
            inputs=None,
            outputs=[transcription_output, status_display, transcribe_btn, stop_btn]
        )
        
        # Add change event for show_prompts checkbox to update display immediately
        show_prompts.change(
            fn=update_transcription_display,
            inputs=[show_prompts],
            outputs=[transcription_output, status_display]
        )
        
        # Set up clear transcriptions button
        clear_transcriptions_btn.click(
            fn=clear_transcriptions,
            inputs=[],
            outputs=[clear_status]
        )
        
        return transcribe_tab, summary_display

def update_summary():
    """Update the summary display with current split state information.
    
    Returns:
        str: Formatted summary of the current split state
    """
    return get_split_summary()

def start_transcription(language, model_path, use_prompt, prompt_type, custom_prompt, show_prompts=True, prompt_trim_type="Calculated (Automatic)", custom_char_limit=300):
    """Start the transcription process.
    
    Args:
        language (str): Language for transcription
        model_path (str): Path to the whisper.cpp model
        use_prompt (bool): Whether to use any prompts at all
        prompt_type (str): Type of prompt to use ("No Prompt", "Previous Chunk", "Custom Prompt")
        custom_prompt (str): Custom prompt text to use for all chunks
        show_prompts (bool): Whether to show prompts in the transcription display
        prompt_trim_type (str): Method for trimming the previous chunk for use as a prompt
        custom_char_limit (int): Custom character limit for the previous chunk
        
    Returns:
        tuple: Status message, transcribe button state, stop button state
    """
    global is_transcribing, transcription_service, transcription_thread, chunk_prompts
    
    # Clear previous chunk prompts
    chunk_prompts = {}
    
    if is_transcribing:
        return "Transcription already in progress", gr.update(interactive=False), gr.update(interactive=True)
    
    # Get the current split state
    try:
        state = get_split_state()
        if not state:
            return "No audio has been split yet. Please go to the Split tab first.", gr.update(interactive=True), gr.update(interactive=False)
    except Exception as e:
        logger.error(f"Error loading split state: {str(e)}")
        return f"Error loading split state: {str(e)}", gr.update(interactive=True), gr.update(interactive=False)
    
    # Sanitize custom prompt if provided
    if use_prompt and prompt_type == "Custom Prompt" and custom_prompt:
        # Basic sanitization - remove control characters only
        custom_prompt = ''.join(c for c in custom_prompt if c.isprintable())
        custom_prompt = custom_prompt.strip()
        # No length limit
        logger.info(f"Using sanitized custom prompt: {custom_prompt}")
    
    # Initialize transcription service
    try:
        transcription_service = TranscriptionService(model_path)
    except Exception as e:
        logger.error(f"Error initializing transcription service: {str(e)}")
        return f"Error initializing transcription service: {str(e)}", gr.update(interactive=True), gr.update(interactive=False)
    
    # Start transcription in a separate thread
    is_transcribing = True
    transcription_thread = threading.Thread(
        target=transcribe_audio_chunks,
        args=(state, language, use_prompt, prompt_type, custom_prompt, show_prompts, prompt_trim_type, custom_char_limit)
    )
    transcription_thread.daemon = True
    transcription_thread.start()
    
    logger.info(f"Started transcription thread for {state['chunks']['total_count']} chunks in {language} (use_prompt={use_prompt}, prompt_type={prompt_type}, show_prompts={show_prompts})")
    
    return f"Transcribing {state['chunks']['total_count']} audio chunks in {language}...", gr.update(interactive=False), gr.update(interactive=True)

def stop_transcription():
    """Stop the transcription process.
    
    Returns:
        tuple: (status message, transcribe button state, stop button state)
    """
    global is_transcribing
    
    if not is_transcribing:
        return "No transcription in progress", gr.update(interactive=True), gr.update(interactive=False)
    
    # Set flag to stop transcription
    is_transcribing = False
    
    # Wait a moment for the transcription thread to finish its current task
    time.sleep(0.5)
    
    # Do one final update of the display with the current state
    # This ensures the final transcription state is displayed
    try:
        # Get the latest transcription content
        content, status = update_transcription_display(True)
        
        # Make sure the status reflects that transcription was stopped
        if "in progress" in status:
            status = status.replace("in progress", "stopped")
    except Exception as e:
        logger.error(f"Error during final transcription display update: {str(e)}")
        content = None
        status = "Transcription stopped. Error displaying final result."
    
    return status, gr.update(interactive=True), gr.update(interactive=False)

def update_transcription_display(show_prompts_enabled=False):
    """Update the transcription display with the latest transcription.
    
    Args:
        show_prompts_enabled (bool): Whether to show prompts in the transcription display
        
    Returns:
        tuple: Transcription text, status message or (None, None) if no update is needed
    """
    global is_transcribing, chunk_prompts
    
    # Handle case when called from Gradio without arguments
    if show_prompts_enabled is None:
        show_prompts_enabled = False
    
    # Only log when actually updating
    if is_transcribing:
        logger.debug(f"Updating transcription display (show_prompts={show_prompts_enabled})")
    
    # Common function to find and process the latest transcription file
    def get_latest_transcription():
        try:
            if os.path.exists(AUDIO_TEXT_DIR):
                files = [f for f in os.listdir(AUDIO_TEXT_DIR) if f.startswith("transcription_") and f.endswith(".txt")]
                if files:
                    # Sort by modification time (newest first)
                    files.sort(key=lambda x: os.path.getmtime(os.path.join(AUDIO_TEXT_DIR, x)), reverse=True)
                    latest_file = os.path.join(AUDIO_TEXT_DIR, files[0])
                    
                    # Read the file content
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Add prompts to display if enabled
                    if show_prompts_enabled and chunk_prompts:
                        # Split content by timestamps
                        parts = []
                        current_chunk = 0
                        
                        # Process the content line by line
                        lines = content.split('\n')
                        i = 0
                        while i < len(lines):
                            line = lines[i]
                            
                            # If this is a timestamp line and we have a prompt for the next chunk
                            if line.startswith('[') and ']' in line and current_chunk in chunk_prompts:
                                prompt = chunk_prompts[current_chunk]
                                if prompt:
                                    # Only show the trimmed version that was actually used
                                    parts.append(f"[Trimmed prompt used for transcription (~{estimate_tokens(prompt['trimmed'])} tokens):]")
                                    trimmed_lines = prompt['trimmed'].split('\n')
                                    for t_line in trimmed_lines:
                                        parts.append(f" {t_line}")
                                    
                                    parts.append("") # Add an empty line after the prompt for better readability
                                
                                # Now add the timestamp line
                                parts.append(line)
                                current_chunk += 1
                            else:
                                # Add any other line
                                parts.append(line)
                            
                            i += 1
                        
                        content = '\n'.join(parts)
                    
                    return content, os.path.basename(latest_file)
        except Exception as e:
            logger.error(f"Error getting latest transcription: {str(e)}")
        
        return "", ""
    
    # Get the latest transcription content
    content, filename = get_latest_transcription()
    
    if not content:
        if is_transcribing:
            return "", "Transcription in progress..."
        else:
            return "", "No transcription available"
    
    # Check if the content contains a completion message
    transcription_completed = "Transcription completed at" in content
    
    # Count only the timestamp lines to estimate progress
    # Look for lines that start with "[" followed by digits and ":" (timestamp format)
    timestamp_count = 0
    for line in content.split('\n'):
        if line.startswith('[') and ']' in line:
            # Check if it's a timestamp line (format: [00:00:00])
            timestamp_part = line.split(']')[0] + ']'
            if re.match(r'\[\d{2}:\d{2}:\d{2}\]', timestamp_part):
                timestamp_count += 1
    
    # Get the total number of chunks from the split state
    try:
        state = get_split_state()
        if state:
            total_chunks = state['chunks']['total_count']
        else:
            total_chunks = timestamp_count
    except Exception as e:
        logger.error(f"Error getting split state: {str(e)}")
        total_chunks = timestamp_count
    
    # Determine the status message based on transcription state
    if transcription_completed or not is_transcribing:
        if transcription_completed:
            status = f"Transcription complete. ({timestamp_count}/{total_chunks} chunks)"
        else:
            status = f"Displaying latest transcription: {filename}"
    else:
        status = f"Transcription in progress... ({timestamp_count}/{total_chunks} chunks)"
    
    return content, status

def transcribe_audio_chunks(state, language, use_prompt=True, prompt_type="Previous Chunk", custom_prompt="", show_prompts=True, prompt_trim_type="Calculated (Automatic)", custom_char_limit=300):
    """Transcribe all audio chunks in the split state.
    
    Args:
        state (dict): The split state
        language (str): Language for transcription
        use_prompt (bool): Whether to use any prompts at all
        prompt_type (str): Type of prompt to use ("No Prompt", "Previous Chunk", "Custom Prompt")
        custom_prompt (str): Custom prompt text to use for all chunks
        show_prompts (bool): Whether to show prompts in the transcription display
        prompt_trim_type (str): Method for trimming the previous chunk for use as a prompt
        custom_char_limit (int): Custom character limit for the previous chunk
    """
    global is_transcribing, transcription_service, chunk_prompts
    
    # Clear previous chunk prompts
    chunk_prompts = {}
    
    logger.info(f"Starting transcription of {state['chunks']['total_count']} audio chunks in {language} (use_prompt={use_prompt}, prompt_type={prompt_type}, show_prompts={show_prompts}, prompt_trim_type={prompt_trim_type}, custom_char_limit={custom_char_limit})")
    
    # Additional sanitization of custom prompt
    if use_prompt and prompt_type == "Custom Prompt" and custom_prompt:
        # Remove any potentially harmful characters
        custom_prompt = ''.join(c for c in custom_prompt if c.isprintable())
        custom_prompt = custom_prompt.strip()
        # Remove the 500 character limit
        logger.info(f"Using sanitized custom prompt: {custom_prompt}")
    
    # Create output directory
    os.makedirs(AUDIO_TEXT_DIR, exist_ok=True)
    
    # Create a combined output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_output_file = os.path.join(AUDIO_TEXT_DIR, f"transcription_{timestamp}.txt")
    
    # Get overlap size from state settings
    overlap_seconds = state['settings'].get('overlap_size', 15)  # Default to 15 seconds if not found
    logger.info(f"Using overlap size of {overlap_seconds} seconds from state settings")
    
    # Clear any existing content in the combined output file
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        f.write(f"Transcription started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original file: {state['original_file']}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Using prompts: {use_prompt}\n")
        if use_prompt:
            f.write(f"Prompt type: {prompt_type}\n")
            if prompt_type == "Custom Prompt":
                # Write the full custom prompt to the file, only sanitizing newlines
                safe_custom_prompt = custom_prompt.replace('\n', ' ').replace('\r', '')
                f.write(f"Custom prompt: {safe_custom_prompt}\n")
            elif prompt_type == "Previous Chunk":
                f.write(f"Prompt trim type: {prompt_trim_type}\n")
                if prompt_trim_type == "Calculated (Automatic)":
                    # Calculate characters based on speaking rate
                    chars_per_second = 2.5 * 5  # 2.5 words/sec * 5 chars/word
                    prompt_chars = int(overlap_seconds * chars_per_second)
                    prompt_tokens = int(prompt_chars / 4)  # 4 chars per token
                    f.write(f"Prompt Length: {overlap_seconds} seconds = ~{prompt_chars} characters ({prompt_tokens} tokens)\n")
                    f.write(f"Maximum Prompt Length: 224 tokens = ~896 characters (179 words)\n")
                else:  # Custom Size
                    f.write(f"Custom Character Limit: {custom_char_limit} characters (~{int(custom_char_limit / 4)} tokens)\n")
                    f.write(f"Maximum Prompt Length: 224 tokens = ~896 characters (179 words)\n")
        
        f.write("\n")
    
    logger.info(f"Created combined output file: {combined_output_file}")
    
    # Get language code
    lang_code = LANGUAGE_MAP.get(language, "en")
    
    # Process each chunk
    previous_text = ""
    total_chunks = state['chunks']['total_count']
    successful_chunks = 0
    
    logger.info(f"Found {total_chunks} chunks to transcribe")
    
    for i, chunk_info in enumerate(state['chunks']['files']):
        if not is_transcribing:
            logger.info("Transcription stopped by user")
            break
        
        chunk_number = chunk_info['chunk_number']
        filename = chunk_info['filename']
        filepath = os.path.join("audio_split", filename)
        
        logger.info(f"Processing chunk {chunk_number}/{total_chunks}: {filename}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"Audio file not found: {filepath}")
            continue
        
        try:
            # Load audio chunk
            logger.debug(f"Loading audio chunk from {filepath}")
            audio_chunk = AudioSegment.from_file(filepath)
            
            # Determine the prompt to use
            initial_prompt = None
            
            # Only use prompts if master toggle is enabled
            if use_prompt:
                if prompt_type == "No Prompt":
                    logger.debug(f"No prompt selected for chunk {chunk_number}")
                elif prompt_type == "Previous Chunk":
                    # Only use previous chunk for chunks after the first one
                    if previous_text and chunk_number > 1:
                        # Determine how to trim the prompt based on the selected method
                        if prompt_trim_type == "Calculated (Automatic)":
                            # Calculate minimum tokens based on overlap time
                            # We want at least 60% of the expected tokens for the overlap period
                            # For a 15-second overlap, this would be ~36 tokens (60% of 60 tokens)
                            min_tokens = max(30, int(overlap_seconds * 2.5 * 1.6 * 0.6))
                            
                            # Trim the previous text to fit within token limit and speaking time
                            # Use the overlap size as the maximum speaking time
                            trimmed_text = trim_text_for_prompt(
                                previous_text, 
                                max_tokens=224, 
                                max_speaking_time=overlap_seconds,
                                min_tokens=min_tokens
                            )
                        else:  # Custom Size
                            # Trim the previous text to the custom character limit
                            trimmed_text = trim_text_to_char_limit(
                                previous_text,
                                char_limit=custom_char_limit
                            )
                        
                        initial_prompt = trimmed_text
                        # Store both the full and trimmed prompts for display purposes
                        chunk_prompts[i] = {
                            'full': previous_text,
                            'trimmed': trimmed_text
                        }
                        
                        # For logging, create a simplified version without line breaks
                        log_prompt = trimmed_text.replace('\n', ' ').replace('\r', '')
                        # Truncate if too long for logs
                        if len(log_prompt) > 100:
                            log_prompt = log_prompt[:97] + "..."
                        
                        # Log both the original and trimmed lengths for debugging
                        original_tokens = estimate_tokens(previous_text)
                        trimmed_tokens = estimate_tokens(trimmed_text)
                        logger.debug(f"Using previous chunk as prompt for chunk {chunk_number} (trimmed from ~{original_tokens} to ~{trimmed_tokens} tokens): {log_prompt}")
                    else:
                        logger.debug(f"No previous chunk available for chunk {chunk_number}")
                        chunk_prompts[i] = None
                elif prompt_type == "Custom Prompt":
                    if custom_prompt:
                        initial_prompt = custom_prompt
                        # Store the prompt for display with consistent format
                        chunk_prompts[i] = {
                            'full': custom_prompt,
                            'trimmed': custom_prompt  # For custom prompts, full and trimmed are the same
                        }
                        logger.debug(f"Using custom prompt for chunk {chunk_number}: {custom_prompt}")
                    else:
                        logger.debug(f"Custom prompt is empty for chunk {chunk_number}")
                        chunk_prompts[i] = None
            else:
                logger.debug(f"Prompts disabled for all chunks")
                chunk_prompts[i] = None
            
            # Transcribe chunk
            logger.info(f"Transcribing chunk {chunk_number} with whisper.cpp")
            transcription = transcription_service.transcribe_chunk(
                audio_chunk,
                lang_code=lang_code,
                initial_prompt=initial_prompt
            )
            
            if not transcription:
                logger.warning(f"No transcription for chunk {chunk_number}")
                continue
            
            logger.info(f"Successfully transcribed chunk {chunk_number}")
            successful_chunks += 1
            
            # Get timecodes if available
            start_time = "00:00:00"
            if chunk_info.get('timecodes') and chunk_info['timecodes'].get('main_segment'):
                start_seconds = chunk_info['timecodes']['main_segment']['start']
                hours = int(start_seconds // 3600)
                minutes = int((start_seconds % 3600) // 60)
                seconds = int(start_seconds % 60)
                start_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Format output with timestamp
            formatted_output = f"[{start_time}] {transcription}\n\n"
            
            # Append to combined output file
            with open(combined_output_file, 'a', encoding='utf-8') as f:
                f.write(formatted_output)
            logger.debug(f"Appended chunk {chunk_number} to combined output file")
            
            # Update previous text for context in next chunk if using previous chunk as prompt
            if use_prompt and prompt_type == "Previous Chunk":
                previous_text = transcription
            else:
                previous_text = ""
            
            logger.info(f"Completed processing chunk {chunk_number}/{total_chunks}")
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk_number}: {str(e)}")
    
    # Write summary at the end of the file
    with open(combined_output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\nTranscription completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Successfully transcribed {successful_chunks} out of {total_chunks} chunks\n")
    
    # Set the global flag to indicate transcription is complete
    is_transcribing = False
    logger.info(f"Transcription complete. Successfully processed {successful_chunks} out of {total_chunks} chunks.")
    
    # Force one final update of the display
    try:
        update_transcription_display(show_prompts)
    except Exception as e:
        logger.error(f"Error during final display update: {str(e)}")
    
    return combined_output_file

def clear_transcriptions():
    """Clear all transcription files from the audio_text directory.
    
    Returns:
        str: Status message indicating the result of the operation
    """
    try:
        # Check if directory exists
        if not os.path.exists(AUDIO_TEXT_DIR):
            logger.warning(f"Transcription directory {AUDIO_TEXT_DIR} does not exist")
            return "No transcriptions found to clear."
        
        # Count files before deletion
        files = [f for f in os.listdir(AUDIO_TEXT_DIR) if f.endswith('.txt')]
        file_count = len(files)
        
        if file_count == 0:
            logger.info(f"No transcription files found in {AUDIO_TEXT_DIR}")
            return "No transcription files found to clear."
        
        # Delete all text files
        for file in files:
            file_path = os.path.join(AUDIO_TEXT_DIR, file)
            try:
                os.remove(file_path)
                logger.debug(f"Deleted transcription file: {file}")
            except Exception as e:
                logger.error(f"Error deleting file {file}: {str(e)}")
                return f"Error while clearing transcriptions: {str(e)}"
        
        logger.info(f"Cleared {file_count} transcription files from {AUDIO_TEXT_DIR}")
        return f"Successfully cleared {file_count} transcription files."
        
    except Exception as e:
        logger.error(f"Error clearing transcriptions: {str(e)}")
        return f"Error clearing transcriptions: {str(e)}"