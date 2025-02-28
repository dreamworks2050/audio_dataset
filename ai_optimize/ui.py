import os
import gradio as gr
from .optimizer import AudioOptimizer
from utils.logger import logger
from pydub import AudioSegment
from transcribe.transcriber import TranscriptionService
import json
import threading
import time
from datetime import datetime

# Initialize the audio optimizer
audio_optimizer = AudioOptimizer()

# Define constants
AUDIO_AI_OPTIMIZED_DIR = "audio_ai_optimized"

# Global variables for transcription
transcription_service = None
is_transcribing = False
transcription_thread = None
stop_transcription = False  # New flag to signal transcription should stop

def create_ai_optimize_tab():
    """Create the AI Optimize tab UI.
    
    Returns:
        gr.TabItem: The AI Optimize tab
    """
    with gr.TabItem("AI Optimize") as ai_optimize_tab:
        with gr.Row():
            # Main content column (2/3 width)
            with gr.Column(scale=2):
                # Audio file selection
                gr.Markdown("### Audio Selection")
                
                # Dropdown for audio files
                audio_files = gr.Dropdown(
                    label="Select Audio File",
                    choices=audio_optimizer.get_audio_files(),
                    type="value",
                    interactive=True
                )
                
                # Refresh button for audio files
                refresh_btn = gr.Button("Refresh Audio Files")
                
                # Add a new row for audio duration selection
                with gr.Row():
                    audio_duration = gr.Radio(
                        label="Audio Duration to Process",
                        choices=["First 5 minutes", "Full Audio"],
                        value="First 5 minutes",
                        interactive=True
                    )
                
                # Chunk and Overlap sections side by side
                with gr.Row():
                    # Left half - Chunk lengths section
                    with gr.Column(scale=1):
                        gr.Markdown("### Chunk Lengths")
                        
                        # Checkboxes for chunk lengths in a single row, including custom
                        with gr.Row():
                            chunk_10s = gr.Checkbox(label="10s", value=True)
                            chunk_15s = gr.Checkbox(label="15s", value=False)
                            chunk_20s = gr.Checkbox(label="20s", value=True)
                            chunk_30s = gr.Checkbox(label="30s", value=True)
                            chunk_45s = gr.Checkbox(label="45s", value=False)
                            chunk_100s = gr.Checkbox(label="100s", value=False)
                            chunk_200s = gr.Checkbox(label="200s", value=False)
                            chunk_custom = gr.Checkbox(label="Custom", value=False)
                            chunk_custom_value = gr.Number(
                                label="",
                                value=25,
                                minimum=1,
                                maximum=600,
                                step=1,
                                interactive=True,
                                visible=False
                            )
                    
                    # Right half - Overlap lengths section
                    with gr.Column(scale=1):
                        gr.Markdown("### Overlap Lengths")
                        
                        # Checkboxes for overlap lengths in a single row, including custom
                        with gr.Row():
                            overlap_0s = gr.Checkbox(label="0s", value=True)
                            overlap_1s = gr.Checkbox(label="1s", value=False)
                            overlap_3s = gr.Checkbox(label="3s", value=True)
                            overlap_5s = gr.Checkbox(label="5s", value=True)
                            overlap_10s = gr.Checkbox(label="10s", value=False)
                            overlap_15s = gr.Checkbox(label="15s", value=False)
                            overlap_20s = gr.Checkbox(label="20s", value=False)
                            overlap_custom = gr.Checkbox(label="Custom", value=False)
                            overlap_custom_value = gr.Number(
                                label="",
                                value=2,
                                minimum=0,
                                maximum=60,
                                step=1,
                                interactive=True,
                                visible=False
                            )
                
                # Buttons row
                with gr.Row():
                    calculate_btn = gr.Button("Calculate Combinations")
                    prepare_btn = gr.Button("Prepare Audios")
                    transcribe_btn = gr.Button("Transcribe Chunks")
                
                # Transcription options - Make visible by default
                with gr.Row(visible=True) as transcription_options:
                    with gr.Column(scale=1):
                        model_path = gr.Dropdown(
                            label="Whisper Model",
                            choices=[
                                "/Users/macbook/audio_dataset/whisper.cpp/models/ggml-large-v3-turbo.bin",
                                "models/ggml-base.en.bin",
                                "models/ggml-small.en.bin",
                                "models/ggml-medium.en.bin",
                                "models/ggml-large.bin"
                            ],
                            value="/Users/macbook/audio_dataset/whisper.cpp/models/ggml-large-v3-turbo.bin",
                            type="value"
                        )
                    
                    with gr.Column(scale=1):
                        language = gr.Dropdown(
                            label="Language",
                            choices=["English", "Korean", "Chinese", "Vietnamese", "Spanish"],
                            value="English",
                            type="value",
                            interactive=True
                        )
                
                # Add prompt options
                with gr.Row(visible=True) as prompt_options:
                    with gr.Column(scale=1):
                        use_prompt = gr.Radio(
                            label="Prompt Settings",
                            choices=["Don't use prompt", "Use Prompt"],
                            value="Don't use prompt",
                            type="value",
                            interactive=True
                        )
                    
                    with gr.Column(scale=1):
                        prompt_length = gr.Number(
                            label="Prompt Length (characters)",
                            value=200,
                            minimum=0,
                            maximum=1000,
                            step=50,
                            interactive=True,
                            visible=False
                        )
                
                # Results display with tabs
                with gr.Row():
                    with gr.Tabs() as result_tabs:
                        with gr.TabItem("Audio Split Results") as audio_split_tab:
                            result_text = gr.Textbox(
                                label="Audio Processing Results",
                                placeholder="Processing results will appear here",
                                lines=15,
                                max_lines=30,
                                interactive=False,
                                show_copy_button=True
                            )
                        
                        with gr.TabItem("Transcription Results") as transcription_tab:
                            with gr.Row():
                                refresh_transcription_btn = gr.Button("Refresh Transcription Status", variant="secondary", scale=1)
                                stop_transcription_btn = gr.Button("Stop Transcription", variant="stop", scale=1)
                            
                            transcription_text = gr.Textbox(
                                label="Transcription Results",
                                placeholder="Transcription results will appear here",
                                lines=15,
                                max_lines=30,
                                interactive=False,
                                show_copy_button=True
                            )
                
                # Status display (small notification area)
                status_display = gr.Markdown("Ready to prepare optimized audio files")
            
            # Summary column (1/3 width)
            with gr.Column(scale=1):
                gr.Markdown("### Processing Summary")
                summary_display = gr.Textbox(
                    label="Combinations Summary",
                    placeholder="Click 'Calculate Combinations' to see what will be processed",
                    lines=20,
                    interactive=False
                )
                
                # Add Clear Folder button
                clear_folder_btn = gr.Button("Clear AI Optimize Folder", variant="secondary")
        
        # Function to update custom value visibility
        def update_custom_visibility(is_checked):
            return gr.update(visible=is_checked)
        
        # Connect custom checkbox to visibility
        chunk_custom.change(
            fn=update_custom_visibility,
            inputs=[chunk_custom],
            outputs=[chunk_custom_value]
        )
        
        overlap_custom.change(
            fn=update_custom_visibility,
            inputs=[overlap_custom],
            outputs=[overlap_custom_value]
        )
        
        # Function to clear the audio_ai_optimized directory
        def clear_ai_optimize_folder():
            """Delete all files and subdirectories in the audio_ai_optimized directory.
            
            Returns:
                str: Status message
            """
            try:
                # Check if directory exists
                if not os.path.exists(AUDIO_AI_OPTIMIZED_DIR):
                    os.makedirs(AUDIO_AI_OPTIMIZED_DIR)
                    return "Directory was empty. Nothing to clear."
                
                # Count items before deletion
                total_items = 0
                for root, dirs, files in os.walk(AUDIO_AI_OPTIMIZED_DIR):
                    total_items += len(files) + len(dirs)
                
                # Delete all files and subdirectories
                for root, dirs, files in os.walk(AUDIO_AI_OPTIMIZED_DIR, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.error(f"Error removing file {file_path}: {str(e)}")
                    
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        try:
                            os.rmdir(dir_path)
                        except Exception as e:
                            logger.error(f"Error removing directory {dir_path}: {str(e)}")
                
                # Keep the main directory but remove all contents
                return f"✅ Successfully cleared {total_items} items from {AUDIO_AI_OPTIMIZED_DIR} directory"
            except Exception as e:
                logger.error(f"Error clearing audio_ai_optimized directory: {str(e)}")
                return f"❌ Error clearing directory: {str(e)}"
        
        # Function to refresh audio files
        def refresh_audio_files():
            return gr.update(choices=audio_optimizer.get_audio_files())
        
        # Connect refresh button
        refresh_btn.click(
            fn=refresh_audio_files,
            inputs=[],
            outputs=[audio_files]
        )
        
        # Update audio files when tab is selected
        ai_optimize_tab.select(
            fn=refresh_audio_files,
            inputs=[],
            outputs=[audio_files]
        )
        
        # Function to validate chunk and overlap calculations
        def validate_chunk_calculations(audio_duration_ms, chunk_length_ms, overlap_ms):
            """Validate the chunk and overlap calculations.
            
            Args:
                audio_duration_ms (int): Duration of the audio in milliseconds
                chunk_length_ms (int): Length of each chunk in milliseconds
                overlap_ms (int): Length of overlap in milliseconds
                
            Returns:
                tuple: (is_valid, total_chunks, validation_message)
            """
            # Check if overlap is less than chunk length
            if overlap_ms >= chunk_length_ms:
                return False, 0, "Overlap must be less than chunk length"
            
            # Calculate effective chunk length (accounting for overlap)
            effective_chunk_length = chunk_length_ms - overlap_ms
            
            # Calculate total chunks
            if effective_chunk_length <= 0:
                return False, 0, "Effective chunk length must be positive"
                
            total_chunks = max(1, int((audio_duration_ms - overlap_ms) / effective_chunk_length) + 1)
            
            # Validate that the chunks will cover the entire audio
            last_chunk_start = (total_chunks - 1) * effective_chunk_length
            last_chunk_end = min(last_chunk_start + chunk_length_ms, audio_duration_ms)
            
            # Check if the last chunk covers the end of the audio
            if last_chunk_end < audio_duration_ms:
                return False, total_chunks, f"Chunks don't cover entire audio (missing {(audio_duration_ms - last_chunk_end)/1000:.2f}s at the end)"
            
            return True, total_chunks, "Valid configuration"
        
        # Function to calculate combinations
        def calculate_combinations(
            audio_file,
            chunk_10s, chunk_15s, chunk_20s, chunk_30s, chunk_45s, chunk_100s, chunk_200s,
            chunk_custom, chunk_custom_value,
            overlap_0s, overlap_1s, overlap_3s, overlap_5s, overlap_10s, overlap_15s, overlap_20s,
            overlap_custom, overlap_custom_value
        ):
            # Validate input
            if not audio_file:
                return "Please select an audio file", "Please select an audio file"
            
            # Collect selected chunk lengths
            chunk_lengths = []
            if chunk_10s: chunk_lengths.append(10)
            if chunk_15s: chunk_lengths.append(15)
            if chunk_20s: chunk_lengths.append(20)
            if chunk_30s: chunk_lengths.append(30)
            if chunk_45s: chunk_lengths.append(45)
            if chunk_100s: chunk_lengths.append(100)
            if chunk_200s: chunk_lengths.append(200)
            if chunk_custom and chunk_custom_value > 0:
                chunk_lengths.append(int(chunk_custom_value))
            
            # Collect selected overlap lengths
            overlap_lengths = []
            if overlap_0s: overlap_lengths.append(0)
            if overlap_1s: overlap_lengths.append(1)
            if overlap_3s: overlap_lengths.append(3)
            if overlap_5s: overlap_lengths.append(5)
            if overlap_10s: overlap_lengths.append(10)
            if overlap_15s: overlap_lengths.append(15)
            if overlap_20s: overlap_lengths.append(20)
            if overlap_custom and overlap_custom_value >= 0:
                overlap_lengths.append(int(overlap_custom_value))
            
            # Validate selections
            if not chunk_lengths:
                return "Please select at least one chunk length", "Please select at least one chunk length"
            if not overlap_lengths:
                return "Please select at least one overlap length", "Please select at least one overlap length"
            
            # Sort the lengths
            chunk_lengths.sort()
            overlap_lengths.sort()
            
            # Try to load the audio file to get its duration
            try:
                audio_path = os.path.join("audio", audio_file)
                audio = AudioSegment.from_file(audio_path)
                audio_duration_ms = len(audio)
                audio_duration_s = audio_duration_ms / 1000
            except Exception as e:
                logger.error(f"Error loading audio file for calculation: {str(e)}")
                error_msg = f"Error loading audio file: {str(e)}"
                return error_msg, error_msg
            
            # Generate summary
            valid_combinations = []
            skipped_combinations = []
            
            # Create summary text
            summary = f"Audio file: {audio_file}\n"
            summary += f"Duration: {audio_duration_s:.2f} seconds\n\n"
            
            # Process each combination
            for chunk_length in chunk_lengths:
                for overlap_length in overlap_lengths:
                    # Skip invalid combinations where overlap >= chunk length
                    if overlap_length >= chunk_length:
                        skipped_combinations.append((chunk_length, overlap_length))
                        continue
                    
                    # Validate the chunk calculations
                    chunk_length_ms = chunk_length * 1000
                    overlap_ms = overlap_length * 1000
                    
                    is_valid, total_chunks, validation_msg = validate_chunk_calculations(
                        audio_duration_ms, chunk_length_ms, overlap_ms
                    )
                    
                    if is_valid:
                        valid_combinations.append((chunk_length, overlap_length, total_chunks, validation_msg))
                    else:
                        skipped_combinations.append((chunk_length, overlap_length, validation_msg))
            
            if valid_combinations:
                summary += f"Valid combinations to process ({len(valid_combinations)}):\n"
                for chunk, overlap, total_chunks, validation_msg in valid_combinations:
                    summary += f"- Chunk: {chunk}s, Overlap: {overlap}s - {total_chunks} chunks (VALID)\n"
                summary += "\n"
            else:
                summary += "No valid combinations to process.\n\n"
            
            if skipped_combinations:
                summary += f"Skipped combinations ({len(skipped_combinations)}):\n"
                for combo in skipped_combinations:
                    if len(combo) == 2:
                        chunk, overlap = combo
                        summary += f"- Chunk: {chunk}s, Overlap: {overlap}s (Overlap must be less than chunk length)\n"
                    else:
                        chunk, overlap, reason = combo
                        summary += f"- Chunk: {chunk}s, Overlap: {overlap}s (Invalid: {reason})\n"
            
            # Add detailed chunk information for each valid combination
            if valid_combinations:
                summary += "\nDetailed Chunk Information:\n"
                for chunk, overlap, total_chunks, _ in valid_combinations:
                    chunk_length_ms = chunk * 1000
                    overlap_ms = overlap * 1000
                    effective_chunk_length = chunk_length_ms - overlap_ms
                    
                    summary += f"\nChunk: {chunk}s, Overlap: {overlap}s - {total_chunks} chunks\n"
                    summary += f"  Effective chunk step: {effective_chunk_length/1000:.2f}s\n"
                    
                    # Calculate and show details for each chunk
                    for i in range(min(total_chunks, 5)):  # Show details for first 5 chunks
                        chunk_start = i * effective_chunk_length
                        chunk_end = min(chunk_start + chunk_length_ms, audio_duration_ms)
                        
                        # Calculate padding before and after
                        padding_before = overlap_ms if i > 0 else 0
                        padding_after = overlap_ms if i < total_chunks - 1 else 0
                        
                        # Calculate actual start and end with padding
                        actual_start = max(0, chunk_start - padding_before)
                        actual_end = min(audio_duration_ms, chunk_end + padding_after)
                        
                        summary += f"  Chunk {i+1}: Base {chunk_start/1000:.2f}s - {chunk_end/1000:.2f}s"
                        
                        if padding_before > 0 or padding_after > 0:
                            summary += f" (with padding: {actual_start/1000:.2f}s - {actual_end/1000:.2f}s, "
                            summary += f"total duration: {(actual_end-actual_start)/1000:.2f}s)\n"
                        else:
                            summary += " (no padding)\n"
                    
                    if total_chunks > 5:
                        summary += f"  ... and {total_chunks - 5} more chunks\n"
            
            # Create a result message for the main display
            result = f"Calculation complete for {audio_file}\n"
            result += f"Found {len(valid_combinations)} valid combinations and {len(skipped_combinations)} invalid combinations.\n"
            result += f"See the summary panel for details."
            
            return summary, result
        
        # Function to prepare optimized audio files
        def prepare_optimized_audio(
            audio_file,
            chunk_10s, chunk_15s, chunk_20s, chunk_30s, chunk_45s, chunk_100s, chunk_200s,
            chunk_custom, chunk_custom_value,
            overlap_0s, overlap_1s, overlap_3s, overlap_5s, overlap_10s, overlap_15s, overlap_20s,
            overlap_custom, overlap_custom_value,
            audio_duration
        ):
            # Validate input
            if not audio_file:
                return "Please select an audio file", "Error: Please select an audio file"
            
            # Collect selected chunk lengths
            chunk_lengths = []
            if chunk_10s: chunk_lengths.append(10)
            if chunk_15s: chunk_lengths.append(15)
            if chunk_20s: chunk_lengths.append(20)
            if chunk_30s: chunk_lengths.append(30)
            if chunk_45s: chunk_lengths.append(45)
            if chunk_100s: chunk_lengths.append(100)
            if chunk_200s: chunk_lengths.append(200)
            if chunk_custom and chunk_custom_value > 0:
                chunk_lengths.append(int(chunk_custom_value))
            
            # Collect selected overlap lengths
            overlap_lengths = []
            if overlap_0s: overlap_lengths.append(0)
            if overlap_1s: overlap_lengths.append(1)
            if overlap_3s: overlap_lengths.append(3)
            if overlap_5s: overlap_lengths.append(5)
            if overlap_10s: overlap_lengths.append(10)
            if overlap_15s: overlap_lengths.append(15)
            if overlap_20s: overlap_lengths.append(20)
            if overlap_custom and overlap_custom_value >= 0:
                overlap_lengths.append(int(overlap_custom_value))
            
            # Validate selections
            if not chunk_lengths:
                return "Please select at least one chunk length", "Error: Please select at least one chunk length"
            if not overlap_lengths:
                return "Please select at least one overlap length", "Error: Please select at least one overlap length"
            
            # Sort the lengths
            chunk_lengths.sort()
            overlap_lengths.sort()
            
            try:
                # Prepare the audio file path
                audio_path = os.path.join("audio", audio_file)
                
                # Start the optimization process
                success, summary = audio_optimizer.optimize_audio(
                    audio_path,
                    chunk_lengths,
                    overlap_lengths,
                    max_duration=None if audio_duration == "Full Audio" else 300  # 5 minutes = 300 seconds
                )
                
                if success:
                    status = "✅ Audio optimization completed successfully"
                    return summary, status
                else:
                    status = "❌ Error during audio optimization"
                    return summary, status
            except Exception as e:
                logger.error(f"Error preparing optimized audio: {str(e)}")
                error_msg = f"Error: {str(e)}"
                return error_msg, "❌ Error during audio optimization"
        
        # Function to transcribe chunks
        def transcribe_chunks(selected_language=None, use_prompt_value=None, prompt_chars=None, latest_run_dir=None):
            global transcription_service, is_transcribing, transcription_thread, stop_transcription
            
            # Check if we're already transcribing
            if is_transcribing:
                return "Transcription is already in progress", "⚠️ Transcription already in progress"
            
            # Reset stop flag when starting a new transcription
            stop_transcription = False
            
            # Use the passed language parameter or fall back to the dropdown value
            if not selected_language:
                selected_language = language.value
                
            if not selected_language:
                return "Please select a language for transcription", "❌ No language selected"
                
            # Log the selected language with extra visibility
            logger.info(f"SELECTED LANGUAGE FOR TRANSCRIPTION: {selected_language} (from direct input)")
            
            # Get prompt settings if not provided
            if use_prompt_value is None:
                use_prompt_value = use_prompt.value
            
            if prompt_chars is None and use_prompt_value == "Use Prompt":
                prompt_chars = prompt_length.value
            
            # Log prompt settings
            if use_prompt_value == "Use Prompt":
                logger.info(f"Using prompt with {prompt_chars} characters from previous chunk")
            else:
                logger.info("Not using prompts for transcription")
            
            # Find the latest run directory if not provided
            if not latest_run_dir:
                if not os.path.exists(AUDIO_AI_OPTIMIZED_DIR):
                    return "No optimized audio files found. Please run 'Prepare Audios' first.", "❌ No optimized audio files found"
                
                # Get all run directories
                run_dirs = [d for d in os.listdir(AUDIO_AI_OPTIMIZED_DIR) 
                           if os.path.isdir(os.path.join(AUDIO_AI_OPTIMIZED_DIR, d)) 
                           and d.startswith("run_")]
                
                if not run_dirs:
                    return "No optimization runs found. Please run 'Prepare Audios' first.", "❌ No optimization runs found"
                
                # Sort by timestamp (newest first)
                run_dirs.sort(reverse=True)
                latest_run_dir = os.path.join(AUDIO_AI_OPTIMIZED_DIR, run_dirs[0])
            
            # Initialize transcription service if needed
            if not transcription_service:
                try:
                    model_path_value = model_path.value
                    transcription_service = TranscriptionService(model_path_value)
                except Exception as e:
                    logger.error(f"Failed to initialize transcription service: {str(e)}")
                    return f"Error initializing transcription service: {str(e)}", "❌ Transcription initialization failed"
            
            # Start transcription in a separate thread
            is_transcribing = True
            transcription_thread = threading.Thread(
                target=transcribe_chunks_thread,
                args=(latest_run_dir, selected_language, use_prompt_value, prompt_chars)
            )
            transcription_thread.daemon = True
            transcription_thread.start()
            
            # Include selected language and prompt info in the status message
            prompt_info = f", With {prompt_chars} character prompts" if use_prompt_value == "Use Prompt" else ""
            status_message = f"🔄 Transcription in progress (Language: {selected_language}{prompt_info})"
            return f"Transcription started with language: {selected_language}{prompt_info}. This may take some time...", status_message
        
        def transcribe_chunks_thread(run_dir, language_name, use_prompt_value="Don't use prompt", prompt_chars=0):
            global is_transcribing, stop_transcription
            
            try:
                # Reset stop flag at the beginning of a new transcription
                stop_transcription = False
                
                # Create a dictionary to track skip reasons
                skip_reasons = {}
                
                # Map language name to code
                language_map = {
                    "Korean": "ko",
                    "English": "en",
                    "Chinese": "zh",
                    "Vietnamese": "vi",
                    "Spanish": "es"
                }
                
                # Get language code and log for debugging
                language_code = language_map.get(language_name, "en")
                language_name_actual = language_name  # Keep original language name for comparison
                
                # Debug log to verify language selection
                logger.warning(f"*** LANGUAGE VERIFICATION ***")
                logger.warning(f"* Selected language name: {language_name_actual}")
                logger.warning(f"* Mapped language code: {language_code}")
                logger.warning(f"* Expected code for Spanish: es")
                
                # Force Spanish if that's what the user selected in UI but somehow got changed
                if language_name.lower() == "spanish":
                    language_code = "es"
                    logger.warning("* Forcing Spanish language code (es) based on language name")
                
                # Log selected language with emphasis and highlighting
                logger.info(f"*** TRANSCRIBING WITH LANGUAGE: {language_name} (CODE: {language_code}) ***")
                logger.debug(f"*** LANGUAGE SELECTION: Using language {language_name} with code {language_code} for transcription ***")
                
                # Log prompt settings
                using_prompts = use_prompt_value == "Use Prompt" and prompt_chars > 0
                if using_prompts:
                    logger.info(f"*** PROMPT SETTINGS: Using {prompt_chars} characters from previous chunk as prompt ***")
                else:
                    logger.info("*** PROMPT SETTINGS: Not using prompts for transcription ***")
                
                # Create a directory for transcriptions and initialize progress file
                transcription_dir = os.path.join(run_dir, "transcriptions")
                os.makedirs(transcription_dir, exist_ok=True)
                
                # Load metadata
                metadata_path = os.path.join(run_dir, "optimization_metadata.json")
                if not os.path.exists(metadata_path):
                    logger.error(f"Metadata file not found: {metadata_path}")
                    return
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Prepare results
                results = []
                total_combinations = len(metadata['combinations'])
                processed_combinations = 0
                
                # Statistics for tracking
                total_chunks = 0
                successful_chunks = 0
                failed_chunks = 0
                skipped_chunks = 0
                
                # Calculate total number of chunks across all combinations
                for combo in metadata['combinations']:
                    total_chunks += len(combo['files'])
                
                # Create/update the progress file to track transcription progress
                progress_path = os.path.join(transcription_dir, "transcription_progress.txt")
                with open(progress_path, 'w', encoding='utf-8') as f:
                    f.write(f"Starting transcription with language: {language_name} (code: {language_code})\n")
                    if using_prompts:
                        f.write(f"Using prompts: Yes, {prompt_chars} characters from previous chunk\n")
                    else:
                        f.write("Using prompts: No\n")
                    f.write(f"Total combinations to process: {total_combinations}\n")
                    f.write(f"Total audio chunks to process: {total_chunks}\n\n")
                
                # Start time for overall process
                transcription_start_time = time.time()
                
                # Process each combination
                for combo in metadata['combinations']:
                    # Check if stop was requested
                    if stop_transcription:
                        logger.warning("Transcription process stopped by user")
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write("\n⚠️ TRANSCRIPTION STOPPED BY USER\n")
                            f.write(f"Stopped at combination {processed_combinations}/{total_combinations}\n")
                        break
                    
                    processed_combinations += 1
                    chunk_length = combo['chunk_length']
                    overlap_length = combo['overlap_length']
                    combo_dir = os.path.join(run_dir, combo['directory'])
                    
                    # Start time for this combination
                    combo_start_time = time.time()
                    
                    # Update progress
                    with open(progress_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n===== Processing combination {processed_combinations}/{total_combinations}: chunk{chunk_length}s_overlap{overlap_length}s =====\n")
                        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    # Create a subdirectory for this combination's transcriptions
                    combo_transcription_dir = os.path.join(transcription_dir, f"chunk{chunk_length}s_overlap{overlap_length}s")
                    os.makedirs(combo_transcription_dir, exist_ok=True)
                    
                    # Process each file in this combination
                    combo_results = []
                    combo_successful = 0
                    combo_failed = 0
                    combo_skipped = 0
                    
                    total_files = len(combo['files'])
                    processed_files = 0
                    
                    # Create a concatenated transcription for this combination
                    all_transcriptions = []
                    
                    # Sort files by chunk number for proper ordering
                    sorted_files = sorted(combo['files'], key=lambda x: x['chunk_number'])
                    
                    # Track the last successful activity time to detect stalls
                    last_activity_time = time.time()
                    
                    # Track previous chunk's transcription for prompts
                    previous_transcription = ""
                    
                    for file_info in sorted_files:
                        # Check if stop was requested
                        if stop_transcription:
                            logger.warning(f"Transcription process stopped by user during combination {processed_combinations}/{total_combinations}")
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"\n⚠️ TRANSCRIPTION STOPPED BY USER\n")
                                f.write(f"Stopped at chunk {processed_files}/{total_files}\n")
                            break
                        
                        # Check for process stalling (no activity for more than 10 minutes)
                        if time.time() - last_activity_time > 600:  # 10 minutes
                            logger.warning(f"Transcription process may be stalled - no activity for 10 minutes")
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"\n⚠️ POSSIBLE STALL DETECTED: No activity for 10 minutes at chunk {processed_files+1}/{total_files}\n")
                        
                        processed_files += 1
                        filename = file_info['filename']
                        file_path = os.path.join(combo_dir, filename)
                        
                        # Start time for this chunk
                        chunk_start_time = time.time()
                        
                        # Update progress for this file
                        chunk_log_message = f"  Processing chunk {processed_files}/{total_files}: {filename}"
                        logger.info(chunk_log_message)
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write(f"{chunk_log_message} (started at {datetime.now().strftime('%H:%M:%S')})\n")
                        
                        # Skip if file doesn't exist
                        if not os.path.exists(file_path):
                            skip_reason = "file_not_found"
                            skip_reason_desc = "Audio file not found"
                            error_msg = f"{skip_reason_desc}: {file_path}"
                            logger.warning(error_msg)
                            skipped_chunks += 1
                            combo_skipped += 1
                            
                            # Track skip reason
                            if skip_reason not in skip_reasons:
                                skip_reasons[skip_reason] = {
                                    "description": skip_reason_desc,
                                    "count": 0,
                                    "chunks": []
                                }
                            skip_reasons[skip_reason]["count"] += 1
                            skip_reasons[skip_reason]["chunks"].append({
                                "chunk_number": file_info["chunk_number"],
                                "filename": filename,
                                "combination": f"chunk{chunk_length}s_overlap{overlap_length}s"
                            })
                            
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"    ❌ SKIPPED: {error_msg}\n")
                                f.write(f"    ℹ️ INFO: This could be due to the file being deleted or never properly created.\n")
                                f.write(f"    🔍 EXPLANATION: Check if the optimization process completed correctly for this combination.\n")
                            continue
                        
                        # Load audio
                        try:
                            audio = AudioSegment.from_file(file_path)
                            
                            # Check if audio is valid (not empty or too short)
                            if len(audio) < 100:  # Less than 100ms is probably invalid
                                skip_reason = "audio_too_short"
                                skip_reason_desc = "Audio file too short"
                                error_msg = f"{skip_reason_desc}: {file_path} ({len(audio)}ms)"
                                logger.warning(error_msg)
                                skipped_chunks += 1
                                combo_skipped += 1
                                
                                # Track skip reason with additional details
                                if skip_reason not in skip_reasons:
                                    skip_reasons[skip_reason] = {
                                        "description": skip_reason_desc,
                                        "count": 0,
                                        "chunks": []
                                    }
                                skip_reasons[skip_reason]["count"] += 1
                                skip_reasons[skip_reason]["chunks"].append({
                                    "chunk_number": file_info["chunk_number"],
                                    "filename": filename,
                                    "duration_ms": len(audio),
                                    "base_start_time": file_info.get("base_start_time", "N/A"),
                                    "base_end_time": file_info.get("base_end_time", "N/A"),
                                    "combination": f"chunk{chunk_length}s_overlap{overlap_length}s"
                                })
                                
                                with open(progress_path, 'a', encoding='utf-8') as f:
                                    f.write(f"    ❌ SKIPPED: {error_msg}\n")
                                    f.write(f"    ℹ️ INFO: This is likely an empty or corrupted audio file.\n")
                                    if file_info["chunk_number"] == len(sorted_files) - 1:
                                        f.write(f"    🔍 EXPLANATION: This is the last chunk in this combination (chunk #{file_info['chunk_number']}), which is often very short or empty when the audio doesn't divide evenly into {chunk_length}s chunks.\n")
                                    else:
                                        f.write(f"    🔍 EXPLANATION: The audio at this position may be silent or corrupted. Check the original audio file at around {file_info.get('base_start_time', 'N/A')}s.\n")
                                continue
                            
                            # Prepare prompt if enabled and not the first chunk
                            initial_prompt = None
                            if using_prompts and previous_transcription and file_info['chunk_number'] > 0:
                                # Get the last N characters, but make sure not to cut words
                                if len(previous_transcription) <= prompt_chars:
                                    initial_prompt = previous_transcription
                                else:
                                    # Start from the last prompt_chars characters
                                    prompt_text = previous_transcription[-prompt_chars:]
                                    # Find the first space to avoid cutting words
                                    first_space = prompt_text.find(' ')
                                    if first_space > 0:
                                        # If there's a space, start from there
                                        initial_prompt = prompt_text[first_space+1:]
                                    else:
                                        # If no space, just use the whole segment
                                        initial_prompt = prompt_text
                                
                                if initial_prompt:
                                    logger.info(f"Using prompt for chunk {file_info['chunk_number']}: '{initial_prompt[:50]}...' ({len(initial_prompt)} chars)")
                                    with open(progress_path, 'a', encoding='utf-8') as f:
                                        f.write(f"    Using prompt: [{len(initial_prompt)} chars] {initial_prompt[:50]}...\n")
                            
                            # Transcribe with error handling
                            try:
                                transcription = transcription_service.transcribe_chunk(audio, lang_code=language_code, initial_prompt=initial_prompt)
                                
                                # Update last activity time since we successfully processed this chunk
                                last_activity_time = time.time()
                                
                                if transcription:
                                    # Save current transcription for next chunk's prompt
                                    previous_transcription = transcription
                                    
                                    # Calculate elapsed time for this chunk
                                    chunk_elapsed_time = time.time() - chunk_start_time
                                    
                                    # Prepare transcription text with prompt information if used
                                    full_transcription = transcription
                                    if initial_prompt:
                                        full_transcription = f"[PROMPT] {initial_prompt}\n\n{transcription}"
                                    
                                    # Save individual transcription to file
                                    txt_filename = os.path.splitext(filename)[0] + ".txt"
                                    txt_path = os.path.join(combo_transcription_dir, txt_filename)
                                    
                                    with open(txt_path, 'w', encoding='utf-8') as f:
                                        # Only save the raw transcription text, without any prompt info
                                        f.write(transcription)
                                    
                                    # Add to concatenated transcriptions
                                    all_transcriptions.append({
                                        'chunk_number': file_info['chunk_number'],
                                        'start_time': file_info['base_start_time'],
                                        'end_time': file_info['base_end_time'],
                                        'text': transcription,
                                        'prompt_used': bool(initial_prompt),
                                        'prompt_text': initial_prompt if initial_prompt else None,
                                        'prompt_length': len(initial_prompt) if initial_prompt else 0
                                    })
                                    
                                    # Add to results
                                    combo_results.append({
                                        'chunk_number': file_info['chunk_number'],
                                        'filename': filename,
                                        'txt_filename': txt_filename,
                                        'base_start_time': file_info['base_start_time'],
                                        'base_end_time': file_info['base_end_time'],
                                        'actual_start_time': file_info['actual_start_time'],
                                        'actual_end_time': file_info['actual_end_time'],
                                        'transcription': transcription,
                                        'prompt_used': bool(initial_prompt),
                                        'prompt_text': initial_prompt if initial_prompt else None,
                                        'elapsed_time': chunk_elapsed_time
                                    })
                                    
                                    # Log success
                                    prompt_info = f" (with prompt)" if initial_prompt else ""
                                    success_msg = f"    ✅ SUCCESS{prompt_info}: Transcribed in {chunk_elapsed_time:.2f}s"
                                    logger.info(f"Successfully transcribed chunk {processed_files}/{total_files} in {chunk_elapsed_time:.2f}s{prompt_info}")
                                    with open(progress_path, 'a', encoding='utf-8') as f:
                                        f.write(f"{success_msg}\n")
                                    
                                    successful_chunks += 1
                                    combo_successful += 1
                                else:
                                    # Transcription failed but didn't raise an exception
                                    error_msg = f"Failed to transcribe (no result returned)"
                                    logger.warning(f"Transcription failed for chunk {processed_files}/{total_files}: {error_msg}")
                                    with open(progress_path, 'a', encoding='utf-8') as f:
                                        f.write(f"    ❌ FAILED: {error_msg} after {time.time() - chunk_start_time:.2f}s\n")
                                    failed_chunks += 1
                                    combo_failed += 1
                            except Exception as e:
                                # Exception during transcription
                                error_msg = f"Error during transcription: {str(e)}"
                                logger.error(error_msg, exc_info=True)
                                with open(progress_path, 'a', encoding='utf-8') as f:
                                    f.write(f"    ❌ ERROR: {error_msg} after {time.time() - chunk_start_time:.2f}s\n")
                                failed_chunks += 1
                                combo_failed += 1
                        except Exception as e:
                            # Exception during audio loading
                            error_msg = f"Failed to process audio file {filename}: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"    ❌ ERROR: {error_msg}\n")
                            failed_chunks += 1
                            combo_failed += 1
                    
                    # Calculate elapsed time for this combination
                    combo_elapsed_time = time.time() - combo_start_time
                    
                    # Log combination completion
                    combo_summary = f"Combination completed in {combo_elapsed_time:.2f}s: {combo_successful} successful, {combo_failed} failed, {combo_skipped} skipped"
                    logger.info(combo_summary)
                    with open(progress_path, 'a', encoding='utf-8') as f:
                        f.write(f"\nCombination summary: {combo_summary}\n")
                    
                    # Save concatenated transcription for this combination
                    if all_transcriptions:
                        concatenated_file_path = os.path.join(combo_dir, f"combined_transcription.txt")
                        concatenated_json_path = os.path.join(combo_dir, f"combined_transcription.json")
                        
                        # Sort by chunk number to ensure correct order
                        all_transcriptions.sort(key=lambda x: x['chunk_number'])
                        
                        # Save as plain text file
                        with open(concatenated_file_path, 'w', encoding='utf-8') as f:
                            for item in all_transcriptions:
                                # Convert seconds to HH:MM:SS format
                                start_hours, start_remainder = divmod(item['start_time'], 3600)
                                start_minutes, start_seconds = divmod(start_remainder, 60)
                                start_time_str = f"{int(start_hours):02d}:{int(start_minutes):02d}:{int(start_seconds):02d}"
                                
                                end_hours, end_remainder = divmod(item['end_time'], 3600)
                                end_minutes, end_seconds = divmod(end_remainder, 60)
                                end_time_str = f"{int(end_hours):02d}:{int(end_minutes):02d}:{int(end_seconds):02d}"
                                
                                time_str = f"[{start_time_str} - {end_time_str}]"
                                
                                # First write prompt if used (before timecode)
                                if item.get('prompt_used') and item.get('prompt_text'):
                                    # Preserve line breaks in the prompt text
                                    f.write(f"[PROMPT] {item['prompt_text']}\n")
                                
                                # Then write timecode followed by transcription text
                                f.write(f"{time_str} {item['text']}\n\n")
                        
                        # Save as JSON for more structured data
                        with open(concatenated_json_path, 'w', encoding='utf-8') as f:
                            json.dump(all_transcriptions, f, indent=2)
                        
                        # Add combination results
                        results.append({
                            'chunk_length': chunk_length,
                            'overlap_length': overlap_length,
                            'directory': combo['directory'],
                            'transcription_directory': os.path.relpath(combo_transcription_dir, run_dir),
                            'concatenated_file': os.path.relpath(concatenated_file_path, run_dir),
                            'concatenated_json': os.path.relpath(concatenated_json_path, run_dir),
                            'files': combo_results,
                            'successful': combo_successful,
                            'failed': combo_failed,
                            'skipped': combo_skipped,
                            'elapsed_time': combo_elapsed_time,
                            'using_prompts': using_prompts,
                            'prompt_length': prompt_chars if using_prompts else 0
                        })
                        
                        # Update progress with concatenated file info
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write(f"  Created combined transcription file: {os.path.basename(concatenated_file_path)}\n")
                    else:
                        # No successful transcriptions for this combination
                        logger.warning(f"No successful transcriptions for combination chunk{chunk_length}s_overlap{overlap_length}s")
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write(f"  ⚠️ No successful transcriptions for this combination, no combined file created\n")
                
                # Calculate total elapsed time
                total_elapsed_time = time.time() - transcription_start_time
                
                # Save transcription metadata
                transcription_metadata = {
                    'original_metadata': metadata_path,
                    'language': language_name,
                    'language_code': language_code,
                    'using_prompts': using_prompts,
                    'prompt_length': prompt_chars if using_prompts else 0,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'combinations': results,
                    'statistics': {
                        'total_chunks': total_chunks,
                        'successful': successful_chunks,
                        'failed': failed_chunks,
                        'skipped': skipped_chunks,
                        'elapsed_time': total_elapsed_time
                    },
                    'skip_reasons': skip_reasons
                }
                
                transcription_metadata_path = os.path.join(transcription_dir, "transcription_metadata.json")
                with open(transcription_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription_metadata, f, indent=2)
                
                # Generate summary
                summary = generate_transcription_summary(transcription_metadata, run_dir)
                
                # Write final summary to progress file
                with open(progress_path, 'a', encoding='utf-8') as f:
                    f.write("\n==============================================\n")
                    if stop_transcription:
                        f.write("⛔️ TRANSCRIPTION STOPPED BY USER ⛔️\n")
                    else:
                        f.write("✅✅✅ TRANSCRIPTION COMPLETE ✅✅✅\n")
                    f.write(f"Total elapsed time: {total_elapsed_time:.2f} seconds\n")
                    f.write(f"Total chunks: {total_chunks}\n")
                    f.write(f"  ✅ Successful: {successful_chunks}\n")
                    f.write(f"  ❌ Failed: {failed_chunks}\n")
                    f.write(f"  ⚠️ Skipped: {skipped_chunks}\n")
                    if total_chunks > 0:
                        f.write(f"Success rate: {successful_chunks/total_chunks*100:.1f}%\n")
                    f.write(f"Created {len(results)} combined transcription files, one for each combination\n")
                
                # We can't directly update the UI from a thread, so we'll write to a file that will be polled
                summary_path = os.path.join(transcription_dir, "transcription_summary.txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                # Create a skipped chunks report if any were skipped
                if skipped_chunks > 0 and 'skip_reasons' in transcription_metadata:
                    skipped_report_path = os.path.join(transcription_dir, "skipped_chunks_report.txt")
                    with open(skipped_report_path, 'w', encoding='utf-8') as f:
                        f.write("===== SKIPPED CHUNKS DETAILED REPORT =====\n\n")
                        f.write(f"Total skipped chunks: {skipped_chunks} out of {total_chunks} ({skipped_chunks/total_chunks*100:.1f}%)\n\n")
                        
                        # Group by reason
                        f.write("=== SKIPPED CHUNKS BY REASON ===\n")
                        for reason_key, reason_data in transcription_metadata['skip_reasons'].items():
                            f.write(f"\n• {reason_data['description']} - {reason_data['count']} chunks\n")
                            
                            # List chunks with this reason
                            for chunk in reason_data['chunks']:
                                combo = chunk['combination']
                                chunk_num = chunk['chunk_number']
                                filename = chunk['filename']
                                
                                chunk_info = f"  - Chunk #{chunk_num} in {combo}: {filename}"
                                
                                # Add duration info if available
                                if 'duration_ms' in chunk:
                                    chunk_info += f" (duration: {chunk['duration_ms']}ms)"
                                
                                # Add time info if available
                                if 'base_start_time' in chunk and 'base_end_time' in chunk:
                                    if chunk['base_start_time'] != "N/A" and chunk['base_end_time'] != "N/A":
                                        chunk_info += f" - audio region {chunk['base_start_time']:.2f}s-{chunk['base_end_time']:.2f}s"
                                
                                f.write(f"{chunk_info}\n")
                            
                        # Group by combination
                        f.write("\n\n=== SKIPPED CHUNKS BY COMBINATION ===\n")
                        combo_skips = {}
                        
                        # Collect chunks by combination
                        for reason_data in transcription_metadata['skip_reasons'].values():
                            for chunk in reason_data['chunks']:
                                combo = chunk['combination']
                                if combo not in combo_skips:
                                    combo_skips[combo] = []
                                    
                                combo_skips[combo].append({
                                    'chunk_number': chunk['chunk_number'],
                                    'reason': reason_data['description'],
                                    'filename': chunk['filename']
                                })
                        
                        # Output chunks by combination
                        for combo, chunks in sorted(combo_skips.items()):
                            f.write(f"\n• {combo} - {len(chunks)} skipped chunks\n")
                            
                            # Sort by chunk number
                            for chunk in sorted(chunks, key=lambda x: x['chunk_number']):
                                f.write(f"  - Chunk #{chunk['chunk_number']}: {chunk['reason']} - {chunk['filename']}\n")
                        
                        # Identify patterns
                        f.write("\n\n=== SKIP PATTERNS DETECTED ===\n")
                        
                        # Check for skips at the end of combinations (common pattern)
                        last_chunk_skips = 0
                        total_combos = len(metadata['combinations'])
                        combos_with_last_chunk_skipped = set()
                        
                        for combo in metadata['combinations']:
                            combo_name = f"chunk{combo['chunk_length']}s_overlap{combo['overlap_length']}s"
                            if combo_name in combo_skips:
                                combo_file_count = len([f for f in combo['files'] if isinstance(f, dict) and 'chunk_number' in f])
                                
                                for skip in combo_skips[combo_name]:
                                    if skip['chunk_number'] == combo_file_count - 1:
                                        last_chunk_skips += 1
                                        combos_with_last_chunk_skipped.add(combo_name)
                        
                        if last_chunk_skips > 0:
                            f.write(f"• Last-chunk skips: {last_chunk_skips} out of {len(combos_with_last_chunk_skipped)} combinations have the last chunk skipped\n")
                            f.write("  This is normal behavior when the audio doesn't divide evenly into the chosen chunk size.\n")
                            f.write("  The last chunk is often very short or empty, which causes it to be skipped.\n")
                            f.write("  ℹ️ Solution: This is expected behavior and doesn't require any action.\n")
                        
                        # Check for chunks with very short duration
                        very_short_chunks = []
                        for reason_data in transcription_metadata['skip_reasons'].values():
                            if reason_data['description'] == "Audio file too short":
                                for chunk in reason_data['chunks']:
                                    if 'duration_ms' in chunk and chunk['duration_ms'] < 50:
                                        very_short_chunks.append(chunk)
                        
                        if very_short_chunks:
                            f.write(f"\n• Very short chunks: {len(very_short_chunks)} chunks are extremely short (less than 50ms)\n")
                            f.write("  This is typically caused by one of the following:\n")
                            f.write("  1. The audio doesn't divide evenly into the chunk size\n")
                            f.write("  2. The original audio has silent portions\n")
                            f.write("  3. The chunk algorithm created some zero-length chunks\n")
                            f.write("  ℹ️ Solution: Try different chunk sizes or check the original audio for silent sections.\n")
                
                logger.info(f"Transcription complete. Processed {total_chunks} chunks with {successful_chunks} successful ({successful_chunks/total_chunks*100:.1f}% success rate)")
                
            except Exception as e:
                logger.error(f"Critical error during transcription process: {str(e)}", exc_info=True)
                # Write error to progress file
                with open(progress_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n❌ CRITICAL ERROR: {str(e)}\n")
                    f.write("Transcription process interrupted. Please check the logs for details.\n")
            finally:
                is_transcribing = False
                stop_transcription = False  # Reset stop flag
        
        def generate_transcription_summary(metadata, run_dir):
            """Generate a summary of the transcription results."""
            summary = "Transcription Results\n\n"
            
            # Add basic info
            summary += f"Language: {metadata['language']} ({metadata['language_code']})\n"
            summary += f"Timestamp: {metadata['timestamp']}\n"
            summary += f"Directory: {os.path.basename(run_dir)}\n"
            
            # Add prompt information
            if metadata.get('using_prompts', False):
                summary += f"Using prompts: Yes, {metadata.get('prompt_length', 0)} characters from previous chunk\n"
            else:
                summary += "Using prompts: No\n"
            
            # Check if transcription was stopped
            was_stopped = stop_transcription
            if was_stopped:
                summary += f"\n⚠️ TRANSCRIPTION WAS STOPPED BY USER\n"
            
            summary += "\n"
            
            # Add statistics
            if 'statistics' in metadata:
                stats = metadata['statistics']
                total_chunks = stats['total_chunks']
                successful = stats['successful']
                failed = stats['failed']
                skipped = stats['skipped']
                elapsed_time = stats['elapsed_time']
                
                summary += f"Transcription Statistics:\n"
                summary += f"- Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n"
                summary += f"- Total chunks: {total_chunks}\n"
                summary += f"  ✅ Successfully transcribed: {successful} ({successful/total_chunks*100:.1f}%)\n"
                summary += f"  ❌ Failed to transcribe: {failed} ({failed/total_chunks*100:.1f}%)\n"
                summary += f"  ⚠️ Skipped chunks: {skipped} ({skipped/total_chunks*100:.1f}%)\n"
                
                # Add more detailed information about skipped chunks in a prominent section
                if skipped > 0:
                    # Always add a header for skipped chunks section
                    summary += f"\n==== SKIPPED CHUNKS INFORMATION ====\n"
                    
                    if 'skip_reasons' in metadata:
                        skip_reasons = metadata['skip_reasons']
                        
                        # Add breakdown by reason
                        summary += f"Skipped chunks by reason:\n"
                        for reason, data in skip_reasons.items():
                            percentage = data['count']/skipped*100
                            summary += f"• {data['description']}: {data['count']} chunks ({percentage:.1f}%)\n"
                        
                        # Check for last chunk skips (common pattern)
                        last_chunk_skips = 0
                        combos_with_last_chunk_skipped = set()
                        
                        for combo in metadata['combinations']:
                            combo_pattern = f"chunk{combo['chunk_length']}s_overlap{combo['overlap_length']}s"
                            combo_file_count = len([f for f in combo['files'] if isinstance(f, dict) and 'chunk_number' in f])
                            
                            # Look through all skip reasons for this combo
                            for reason_data in skip_reasons.values():
                                for chunk in reason_data['chunks']:
                                    if chunk.get('combination') == combo_pattern and chunk['chunk_number'] == combo_file_count - 1:
                                        last_chunk_skips += 1
                                        combos_with_last_chunk_skipped.add(combo_pattern)
                        
                        # If we detected last chunk skips, highlight this common pattern
                        if last_chunk_skips > 0:
                            summary += f"\n⚠️ Common Pattern Detected: {last_chunk_skips} combinations have their last chunk skipped\n"
                            summary += f"  This is normal behavior when the audio doesn't divide evenly into chunks.\n"
                            summary += f"  The last chunk in each combination is often very short or empty.\n"
                            
                        # Add info about skipped chunks report
                        summary += f"\nA detailed report of all skipped chunks has been generated at:\n"
                        summary += f"- {os.path.join(run_dir, 'transcriptions', 'skipped_chunks_report.txt')}\n"
                    else:
                        # Generic message if we don't have detailed skip reasons
                        summary += f"\nSome chunks were skipped, most likely because they were empty or too short.\n"
                        summary += f"This commonly happens with the last chunk in a combination when the audio doesn't divide evenly.\n"
                    
                    # Add a horizontal line to separate this section
                    summary += f"\n" + "-" * 50 + "\n"
                
                if successful > 0:
                    avg_time_per_chunk = elapsed_time / successful
                    summary += f"- Average time per chunk: {avg_time_per_chunk:.2f} seconds\n"
            else:
                # Count total files (for backward compatibility)
                total_files = sum(len(combo['files']) for combo in metadata['combinations'])
                
                summary += f"Transcribed {total_files} audio chunks across {len(metadata['combinations'])} combinations\n"
            
            summary += f"Created {len(metadata['combinations'])} combined transcription files, one for each combination\n\n"
            
            # Add details for each combination
            summary += "Combination Details:\n"
            for combo in metadata['combinations']:
                chunk_length = combo['chunk_length']
                overlap_length = combo['overlap_length']
                
                summary += f"\n=== Chunk {chunk_length}s, Overlap {overlap_length}s ===\n"
                
                # Add statistics if available
                if 'successful' in combo:
                    total_combo_chunks = combo['successful'] + combo['failed'] + combo['skipped']
                    summary += f"Files: {total_combo_chunks} chunks (✅ {combo['successful']} successful, "
                    summary += f"❌ {combo['failed']} failed, ⚠️ {combo['skipped']} skipped)\n"
                    
                    # Add detailed note about skipped files in this combo
                    if combo['skipped'] > 0:
                        summary += f"Note: {combo['skipped']} chunk(s) skipped in this combination.\n"
                        
                        # Try to find which chunks were skipped in this combination
                        if 'skip_reasons' in metadata:
                            combo_pattern = f"chunk{chunk_length}s_overlap{overlap_length}s"
                            skipped_in_combo = []
                            
                            for reason, data in metadata['skip_reasons'].items():
                                for chunk in data['chunks']:
                                    if chunk.get('combination') == combo_pattern:
                                        skipped_in_combo.append({
                                            'chunk_number': chunk['chunk_number'],
                                            'reason': data['description']
                                        })
                            
                            if skipped_in_combo:
                                summary += "  Skipped chunks in this combination:\n"
                                for skip_info in sorted(skipped_in_combo, key=lambda x: x['chunk_number']):
                                    summary += f"    - Chunk #{skip_info['chunk_number']}: {skip_info['reason']}\n"
                                    
                                # If last chunk was skipped, add explanation
                                last_chunks = [s for s in skipped_in_combo if s['chunk_number'] == total_combo_chunks - 1]
                                if last_chunks:
                                    summary += f"  Note: The last chunk was skipped, likely because the audio doesn't divide evenly into {chunk_length}s chunks.\n"
                    
                    if 'elapsed_time' in combo:
                        summary += f"Processing time: {combo['elapsed_time']:.2f} seconds\n"
                else:
                    summary += f"Files: {len(combo['files'])} chunks\n"
                
                # Add info about the combined transcription file
                if 'concatenated_file' in combo:
                    summary += f"Combined transcription: {os.path.join(run_dir, combo['concatenated_file'])}\n"
                
                # Show sample transcriptions (first 2 chunks)
                if combo['files']:
                    summary += "Sample transcriptions:\n"
                    for i, file_info in enumerate(sorted(combo['files'], key=lambda x: x['chunk_number'])[:2]):
                        summary += f"  Chunk {file_info['chunk_number']}: "
                        # Truncate long transcriptions
                        transcription = file_info['transcription']
                        if len(transcription) > 100:
                            transcription = transcription[:97] + "..."
                        summary += f"{transcription}\n"
                    
                    if len(combo['files']) > 2:
                        summary += f"  ... and {len(combo['files']) - 2} more chunks\n"
            
            # Add path to transcription files
            summary += f"\nTranscription Files Location:\n"
            summary += f"- Base directory: {os.path.join(run_dir, 'transcriptions')}\n"
            summary += f"- Each combination has its own subdirectory with individual .txt files for each audio chunk\n"
            summary += f"- Combined transcription files (one per combination) are saved both in each combination's directory\n"
            summary += f"  and in the transcriptions directory\n"
            summary += f"- Full JSON metadata available at: {os.path.join(run_dir, 'transcriptions', 'transcription_metadata.json')}\n"
            
            # Add troubleshooting info
            if failed > 0 or skipped > 0:
                summary += f"\nTroubleshooting Info:\n"
                if skipped > 0:
                    summary += f"- Skipped chunks are usually due to empty or very short audio files (less than 100ms)\n"
                    summary += f"- This is normal for the last chunk in a combination if the audio doesn't divide evenly\n"
                    
                    # Add common solutions for skipped chunks
                    summary += f"- Common solutions for excessive skipped chunks:\n"
                    summary += f"  1. Try different chunk lengths that divide more evenly into your audio length\n"
                    summary += f"  2. Check the original audio file for silent sections or corrupted segments\n"
                    summary += f"  3. For last-chunk skips (most common), this is normal and can be ignored\n"
                    
                summary += f"- Check the transcription_progress.txt file for detailed error messages and info about skipped chunks\n"
                summary += f"- Progress file: {os.path.join(run_dir, 'transcriptions', 'transcription_progress.txt')}\n"
                summary += f"- See application logs for additional details on failures\n"
            
            return summary
        
        # Function to check transcription progress
        def check_transcription_progress(dummy=None):
            """Check the progress of the transcription process.
            
            Args:
                dummy: Dummy parameter to satisfy Gradio's event handler requirements
            """
            # Find the latest run directory
            run_dirs = [d for d in os.listdir(AUDIO_AI_OPTIMIZED_DIR) 
                       if os.path.isdir(os.path.join(AUDIO_AI_OPTIMIZED_DIR, d)) 
                       and d.startswith("run_")]
            
            if not run_dirs:
                return None
            
            # Sort by timestamp (newest first)
            run_dirs.sort(reverse=True)
            latest_run_dir = os.path.join(AUDIO_AI_OPTIMIZED_DIR, run_dirs[0])
            
            # Check for summary file first (indicates completion)
            summary_path = os.path.join(latest_run_dir, "transcriptions", "transcription_summary.txt")
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = f.read()
                
                # Add a clear completion message at the top if it's not already there
                if "✅ TRANSCRIPTION COMPLETED SUCCESSFULLY ✅" not in summary:
                    # Check if we have statistics to determine if it was a successful completion
                    if "Transcription Statistics:" in summary and not is_transcribing:
                        summary = "✅ TRANSCRIPTION COMPLETED SUCCESSFULLY ✅\n\n" + summary
                
                # Check if there's a skipped chunks report
                skipped_report_path = os.path.join(latest_run_dir, "transcriptions", "skipped_chunks_report.txt")
                if os.path.exists(skipped_report_path) and "Skipped chunks:" in summary:
                    # Extract skipped count from summary
                    try:
                        skipped_count = 0
                        for line in summary.split('\n'):
                            if "⚠️ Skipped chunks:" in line:
                                parts = line.split(':')
                                if len(parts) > 1:
                                    parts = parts[1].strip().split(' ')
                                    if len(parts) > 0:
                                        skipped_count = int(parts[0])
                                break
                        
                        if skipped_count > 0:
                            # Add a notice about the skipped chunks report
                            # Only add this if it doesn't already exist
                            if "DETAILED SKIPPED CHUNKS REPORT AVAILABLE" not in summary:
                                report_notice = f"\n⚠️ DETAILED SKIPPED CHUNKS REPORT AVAILABLE ⚠️\n"
                                report_notice += f"{skipped_count} chunks were skipped during transcription.\n"
                                report_notice += f"See: {os.path.basename(skipped_report_path)} for complete details about each skipped chunk.\n\n"
                                
                                # Insert after the statistics section
                                if "Transcription Statistics:" in summary:
                                    parts = summary.split("Transcription Statistics:")
                                    if len(parts) > 1:
                                        stats_section_end = parts[1].find("\nCreated ")
                                        if stats_section_end > 0:
                                            summary = parts[0] + "Transcription Statistics:" + parts[1][:stats_section_end] + report_notice + parts[1][stats_section_end:]
                                        else:
                                            summary += report_notice
                                    else:
                                        summary += report_notice
                    except Exception as e:
                        logger.error(f"Error adding skipped chunks report notice: {str(e)}")
                
                return summary
            
            # If no summary, check for progress file
            progress_path = os.path.join(latest_run_dir, "transcriptions", "transcription_progress.txt")
            if os.path.exists(progress_path):
                with open(progress_path, 'r', encoding='utf-8') as f:
                    progress = f.read()
                    
                # Check if transcription is complete based on the content
                if "TRANSCRIPTION COMPLETE" in progress and not is_transcribing:
                    progress = "✅ TRANSCRIPTION COMPLETED SUCCESSFULLY ✅\n\n" + progress
                elif "TRANSCRIPTION STOPPED BY USER" in progress and not is_transcribing:
                    progress = "⛔️ TRANSCRIPTION WAS STOPPED BY USER ⛔️\n\n" + progress
                elif is_transcribing:
                    progress = "🔄 TRANSCRIPTION IN PROGRESS 🔄\n\n" + progress
                
                return progress
            
            # If we're still transcribing but no files exist yet
            if is_transcribing:
                return "🔄 TRANSCRIPTION IN PROGRESS - Initializing transcription process... 🔄"
            
            return None
        
        # Function to stop transcription
        def stop_transcription_process():
            """Signal the transcription process to stop.
            
            Returns:
                str: Status message
            """
            global stop_transcription, is_transcribing
            
            if not is_transcribing:
                return "No transcription is currently running."
            
            stop_transcription = True
            logger.warning("User requested to stop transcription process")
            return "⚠️ Stopping transcription... This may take a moment to complete."
        
        # Connect transcribe button
        transcribe_btn.click(
            fn=transcribe_chunks,
            inputs=[language, use_prompt, prompt_length],
            outputs=[transcription_text, status_display]
        )
        
        # Connect refresh transcription button
        refresh_transcription_btn.click(
            fn=check_transcription_progress,
            inputs=None,
            outputs=transcription_text
        )
        
        # Set up polling for transcription progress
        transcription_tab.select(
            fn=check_transcription_progress,
            inputs=None,
            outputs=transcription_text
        )
        
        # Connect calculate button
        calculate_btn.click(
            fn=calculate_combinations,
            inputs=[
                audio_files,
                chunk_10s, chunk_15s, chunk_20s, chunk_30s, chunk_45s, chunk_100s, chunk_200s,
                chunk_custom, chunk_custom_value,
                overlap_0s, overlap_1s, overlap_3s, overlap_5s, overlap_10s, overlap_15s, overlap_20s,
                overlap_custom, overlap_custom_value
            ],
            outputs=[summary_display, result_text]
        )
        
        # Connect prepare button
        prepare_btn.click(
            fn=prepare_optimized_audio,
            inputs=[
                audio_files,
                chunk_10s, chunk_15s, chunk_20s, chunk_30s, chunk_45s, chunk_100s, chunk_200s,
                chunk_custom, chunk_custom_value,
                overlap_0s, overlap_1s, overlap_3s, overlap_5s, overlap_10s, overlap_15s, overlap_20s,
                overlap_custom, overlap_custom_value,
                audio_duration
            ],
            outputs=[result_text, status_display]
        )
        
        # Connect clear folder button
        clear_folder_btn.click(
            fn=clear_ai_optimize_folder,
            inputs=[],
            outputs=[status_display]
        )
        
        # Connect stop transcription button
        stop_transcription_btn.click(
            fn=stop_transcription_process,
            inputs=[],
            outputs=[status_display]
        )
        
        # Connect use_prompt to show/hide prompt_length
        use_prompt.change(
            fn=lambda x: gr.update(visible=(x == "Use Prompt")),
            inputs=[use_prompt],
            outputs=[prompt_length]
        )
    
    return ai_optimize_tab 