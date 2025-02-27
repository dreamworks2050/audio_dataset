import os
import gradio as gr
import threading
import json
from datetime import datetime
from pydub import AudioSegment
from split.state import get_split_summary, get_split_state
from utils.logger import logger
from .transcriber import TranscriptionService

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
        
        # Set up event handlers
        transcribe_tab.select(fn=update_summary, inputs=[], outputs=[summary_display])
        refresh_btn.click(fn=update_summary, inputs=[], outputs=[summary_display])
        
        transcribe_btn.click(
            fn=start_transcription,
            inputs=[language, model_path],
            outputs=[status_display, transcribe_btn, stop_btn]
        )
        
        stop_btn.click(
            fn=stop_transcription,
            inputs=[],
            outputs=[status_display, transcribe_btn, stop_btn]
        )
        
        # Set up periodic refresh of transcription output (every 2 seconds)
        refresh_timer = gr.Timer(value=2)
        refresh_timer.tick(
            fn=update_transcription_display, 
            inputs=[], 
            outputs=[transcription_output, status_display]
        )
        
        return transcribe_tab, summary_display

def update_summary():
    """Update the summary display with current split state information.
    
    Returns:
        str: Formatted summary of the current split state
    """
    return get_split_summary()

def start_transcription(language, model_path):
    """Start the transcription process.
    
    Args:
        language (str): Language for transcription
        model_path (str): Path to the whisper.cpp model
        
    Returns:
        tuple: Status message, transcribe button state, stop button state
    """
    global is_transcribing, transcription_service, transcription_thread
    
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
        args=(state, language)
    )
    transcription_thread.daemon = True
    transcription_thread.start()
    
    logger.info(f"Started transcription thread for {state['chunks']['total_count']} chunks in {language}")
    
    return f"Transcribing {state['chunks']['total_count']} audio chunks in {language}...", gr.update(interactive=False), gr.update(interactive=True)

def stop_transcription():
    """Stop the transcription process.
    
    Returns:
        tuple: (status message, transcribe button state, stop button state)
    """
    global is_transcribing
    
    if not is_transcribing:
        return "No transcription in progress", gr.update(interactive=True), gr.update(interactive=False)
    
    is_transcribing = False
    return "Transcription stopping...", gr.update(interactive=True), gr.update(interactive=False)

def update_transcription_display():
    """Update the transcription display with the latest transcription.
    
    Returns:
        tuple: Transcription text, status message
    """
    global is_transcribing
    
    # Check if transcription is in progress
    if is_transcribing:
        status = "Transcription in progress..."
        
        # Find the latest transcription file
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
                    
                    # Count the number of timestamps to estimate progress
                    timestamp_count = content.count("[")
                    
                    # Get the total number of chunks from the split state
                    try:
                        state = get_split_state()
                        if state:
                            total_chunks = state['chunks']['total_count']
                            status = f"Transcription in progress... ({timestamp_count}/{total_chunks} chunks)"
                    except Exception as e:
                        logger.error(f"Error getting split state: {str(e)}")
                    
                    return content, status
        except Exception as e:
            logger.error(f"Error updating transcription display: {str(e)}")
    else:
        status = "Transcription not running"
        
        # Find the latest transcription file
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
                    
                    status = f"Displaying latest transcription: {os.path.basename(latest_file)}"
                    return content, status
        except Exception as e:
            logger.error(f"Error updating transcription display: {str(e)}")
    
    return "", status

def transcribe_audio_chunks(state, language):
    """Transcribe all audio chunks in the split state.
    
    Args:
        state (dict): The split state
        language (str): Language for transcription
    """
    global is_transcribing, transcription_service
    
    logger.info(f"Starting transcription of {state['chunks']['total_count']} audio chunks in {language}")
    
    # Create output directory
    os.makedirs(AUDIO_TEXT_DIR, exist_ok=True)
    
    # Create a combined output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_output_file = os.path.join(AUDIO_TEXT_DIR, f"transcription_{timestamp}.txt")
    
    # Clear any existing content in the combined output file
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        f.write(f"Transcription started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original file: {state['original_file']}\n")
        f.write(f"Language: {language}\n\n")
    
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
            
            # Create initial prompt from previous text if available
            initial_prompt = ""
            if previous_text:
                # Use the last 30 words as context
                words = previous_text.split()
                context = " ".join(words[-30:]) if len(words) > 30 else previous_text
                initial_prompt = context
                logger.debug(f"Using initial prompt from previous chunk: {context[:50]}...")
            
            # Transcribe chunk
            logger.info(f"Transcribing chunk {chunk_number} with whisper.cpp")
            transcription = transcription_service.transcribe_chunk(
                audio_chunk,
                lang_code=lang_code,
                initial_prompt=initial_prompt if initial_prompt else None
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
            
            # Update previous text for context in next chunk
            previous_text = transcription
            
            logger.info(f"Completed processing chunk {chunk_number}/{total_chunks}")
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk_number}: {str(e)}", exc_info=True)
    
    # Write summary at the end of the file
    with open(combined_output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\nTranscription completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Successfully transcribed {successful_chunks} out of {total_chunks} chunks\n")
    
    is_transcribing = False
    logger.info(f"Transcription complete. Successfully processed {successful_chunks} out of {total_chunks} chunks.")
    
    return combined_output_file