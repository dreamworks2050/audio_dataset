import gradio as gr
from datetime import datetime
import json
from download.downloader import get_youtube_metadata, download_video, download_audio
import os
from split.split import get_audio_files, split_audio
from utils.cleanup import cleanup_python_cache
from utils.logger import logger

# Clear logs at startup
logger.info("Starting application")
logger.clear_logs()
logger.info("Application started - logs cleared")

# Save metadata to a file
def save_metadata(metadata):
    """Save metadata to a JSON file in the 'metadata' directory."""
    os.makedirs("metadata", exist_ok=True)
    filename = "metadata/metadata_01.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(metadata, indent=2))
    return filename

# Summarize metadata and prepare format information
def summarize_metadata(metadata):
    """Process metadata and extract relevant information for display and downloads."""
    # Extract basic info from root-level properties
    title = metadata.get('title', 'Unknown Title')
    duration = metadata.get('duration', 0)  # Fallback to 0 if missing
    upload_date = metadata.get('upload_date', None)
    
    # Format duration (convert seconds to MM:SS)
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    duration_str = f"{minutes}m {seconds}s"
    
    # Format upload date (convert YYYYMMDD to readable format)
    upload_date_str = (
        datetime.strptime(upload_date, "%Y%m%d").strftime("%B %d, %Y")
        if upload_date else "Unknown Upload Date"
    )
    
    # Filter and sort video formats according to schema requirements
    video_formats = [
        fmt for fmt in metadata.get('formats', [])
        if fmt.get('vcodec', '').lower() != 'none' and 
           isinstance(fmt.get('height'), int) and
           isinstance(fmt.get('width'), int)
    ]
    
    # Sort video formats by height (resolution) and select the highest quality
    video_formats.sort(key=lambda x: (x.get('height', 0), x.get('width', 0)), reverse=True)
    selected_video_format = video_formats[0] if video_formats else None
    
    video_format = "No video available"
    if selected_video_format:
        height = selected_video_format.get('height')
        vcodec = selected_video_format.get('vcodec', 'unknown').lower()
        video_format = f"{height}p ({height}p) - {vcodec}"
    
    # Filter and sort audio formats according to schema requirements
    audio_formats = [
        fmt for fmt in metadata.get('formats', [])
        if fmt.get('vcodec', '').lower() == 'none' and 
           fmt.get('acodec', '').lower() != 'none' and 
           isinstance(fmt.get('abr'), (int, float))
    ]
    
    # Sort audio formats by bitrate and select the highest quality
    audio_formats.sort(key=lambda x: float(x.get('abr', 0)), reverse=True)
    selected_audio_format = audio_formats[0] if audio_formats else None
    
    audio_format = "No audio available"
    if selected_audio_format:
        abr = selected_audio_format.get('abr', 0)
        acodec = selected_audio_format.get('acodec', 'unknown').lower()
        audio_format = f"{int(abr)} kbps (medium) - {acodec}"
    
    # Build comprehensive format lists for display
    video_format_list = []
    for fmt in sorted(video_formats, key=lambda x: (-x.get('height', 0), -x.get('width', 0))):
        height = fmt.get('height')
        vcodec = fmt.get('vcodec', 'unknown').lower()
        is_known = any(known in vcodec for known in ['avc1', 'vp9', 'vp09'])
        format_note = fmt.get('format_note', str(height) + 'p')
        
        if is_known:
            format_str = f"- {height}p ({format_note}) - {vcodec}"
        else:
            format_str = f"- {height}p (unknown) - {vcodec}"
        video_format_list.append(format_str)
    
    audio_format_list = []
    for fmt in sorted(audio_formats, key=lambda x: -float(x.get('abr', 0))):
        abr = fmt.get('abr', 0)
        acodec = fmt.get('acodec', 'unknown').lower()
        format_note = fmt.get('format_note', 'medium')
        format_str = f"- {int(abr)} kbps ({format_note}) - {acodec}"
        audio_format_list.append(format_str)
    
    # Build summary text with all formats
    summary = (
        f"**Title:** {title}\n\n"
        f"**Duration:** {duration_str}\n\n"
        f"**Upload Date:** {upload_date_str}\n\n"
        f"**Available Video Formats:**\n\n"
        f"{chr(10).join(video_format_list)}\n\n"
        f"**Available Audio Formats:**\n\n"
        f"{chr(10).join(audio_format_list)}"
    )
    
    return summary, video_format, audio_format

# Handle metadata retrieval and UI updates
def grab_metadata(url):
    """Fetch metadata and update Gradio components."""
    if not url:
        return "Please enter a URL", ""
    try:
        metadata = get_youtube_metadata(url)
        save_metadata(metadata)
        summary, video_format, audio_format = summarize_metadata(metadata)
        selected_formats = (
            f"Selected formats for download:\n\n"
            f"Download Video: {video_format}\n"
            f"Download Audio: {audio_format}"
        )
        return summary, selected_formats
    except Exception as e:
        return f"Error: {str(e)}", ""

# Clean up Python cache files before starting
cleanup_python_cache()

# Build Gradio interface
with gr.Blocks() as ui:
    with gr.Tabs():
        # Import and create the AI Optimize tab (at the top)
        from ai_optimize import create_ai_optimize_tab
        create_ai_optimize_tab()
        
        with gr.TabItem("Download"):
            with gr.Row():
                with gr.Column(scale=2):
                    url_input = gr.Textbox(label="YouTube URL", placeholder="Paste YouTube URL here")
                    metadata_display = gr.Textbox(label="Metadata Summary", lines=10, interactive=False)
                    format_display = gr.Textbox(label="Selected Formats", interactive=False)
                    status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    grab_info_btn = gr.Button("Grab Information")
                    download_video_btn = gr.Button("Download Video")
                    download_audio_btn = gr.Button("Download Audio")
            
            # Connect buttons to functions
            grab_info_btn.click(
                fn=grab_metadata,
                inputs=[url_input],
                outputs=[metadata_display, format_display]
            )
            download_video_btn.click(
                fn=download_video,
                inputs=[url_input],
                outputs=[status]
            )
            download_audio_btn.click(
                fn=download_audio,
                inputs=[url_input],
                outputs=[status]
            )
        
        with gr.TabItem("Split Audio", id="split_audio_tab") as split_audio_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(label="Upload Audio File", type="filepath")
                with gr.Column(scale=1):
                    audio_files = gr.Dropdown(
                        label="Select Audio File",
                        choices=get_audio_files(),
                        type="value",
                        interactive=True
                    )

            def update_audio_files():
                return gr.update(choices=get_audio_files())

            split_audio_tab.select(fn=update_audio_files, inputs=[], outputs=[audio_files])
            with gr.Row():
                with gr.Column():
                    split_size = gr.Dropdown(
                        label="Split Size",
                        choices=["10s", "30s", "45s", "60s", "100s", "Custom"],
                        type="value",
                        interactive=True
                    )
                    custom_split_size = gr.Textbox(
                        label="Custom Split Size (in seconds)",
                        visible=False
                    )
                    
                    overlap_size = gr.Dropdown(
                        label="Overlap Duration",
                        choices=["1s", "5s", "10s", "15s", "20s", "25s", "30s", "45s", "Custom", "None"],
                        type="value",
                        value="None",
                        interactive=True
                    )
                    custom_overlap_size = gr.Textbox(
                        label="Custom Overlap Duration (in seconds)",
                        visible=False
                    )
                    
                    def update_visibility(split_choice, overlap_choice):
                        return [
                            gr.update(visible=split_choice == "Custom"),
                            gr.update(visible=overlap_choice == "Custom")
                        ]
                    
                    split_size.change(fn=update_visibility, inputs=[split_size, overlap_size], outputs=[custom_split_size, custom_overlap_size])
                    overlap_size.change(fn=update_visibility, inputs=[split_size, overlap_size], outputs=[custom_split_size, custom_overlap_size])
                    
                    split_btn = gr.Button("Split Audio")
                    split_status = gr.Textbox(label="Split Status", interactive=False)
                    
                    def process_split(audio_path, size_choice, custom_size, overlap_choice, custom_overlap):
                        try:
                            # Handle both uploaded files and selected files from dropdown
                            if not audio_path and not audio_files.value:
                                return "Please select or upload an audio file"
                            
                            # Use the selected file from dropdown if no upload
                            if not audio_path and audio_files.value:
                                audio_path = os.path.join("audio", audio_files.value)
                            
                            # Convert size choice to seconds
                            if size_choice == "Custom":
                                if not custom_size or not custom_size.isdigit():
                                    return "Please enter a valid number of seconds"
                                split_size = int(custom_size)
                            else:
                                split_size = int(size_choice.rstrip("s"))
                            
                            # Convert overlap choice to seconds
                            overlap_size = 0
                            if overlap_choice != "None":
                                if overlap_choice == "Custom":
                                    if not custom_overlap or not custom_overlap.isdigit():
                                        return "Please enter a valid overlap duration in seconds"
                                    overlap_size = int(custom_overlap)
                                else:
                                    overlap_size = int(overlap_choice.rstrip("s"))
                            
                            # Perform the split
                            split_files, summary = split_audio(audio_path, split_size, overlap_size)
                            return summary
                        except Exception as e:
                            return f"Error: {str(e)}"
                    
                    split_btn.click(
                        fn=process_split,
                        inputs=[audio_input, split_size, custom_split_size, overlap_size, custom_overlap_size],
                        outputs=[split_status]
                    )
                    
                    # Refresh audio files list when new file is uploaded
                    audio_input.change(
                        fn=lambda: gr.update(choices=get_audio_files()),
                        outputs=[audio_files]
                    )
        
        from transcribe import create_transcribe_tab
        create_transcribe_tab()

# Launch the interface
ui.launch()