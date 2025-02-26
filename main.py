import gradio as gr
from datetime import datetime
import json
from download.downloader import get_youtube_metadata, download_video, download_audio
import os
from split.split import get_audio_files, split_audio, process_split, update_custom_visibility
from utils.logger import logger
import signal
import sys
from utils.logger import Logger

# Initialize logger
logger = Logger()

def signal_handler(sig, frame):
    """Handle keyboard interrupt (Ctrl+C) gracefully."""
    logger.info('\nReceived keyboard interrupt (Ctrl+C). Performing cleanup...')
    # Add any cleanup operations here
    logger.info('Cleanup complete. Exiting gracefully.')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

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
        logger.warning("No URL provided")
        return "Please enter a URL", ""
    try:
        logger.info(f"Fetching metadata for URL: {url}")
        metadata = get_youtube_metadata(url)
        logger.debug("Saving metadata to file")
        save_metadata(metadata)
        logger.debug("Processing metadata summary")
        summary, video_format, audio_format = summarize_metadata(metadata)
        selected_formats = (
            f"Selected formats for download:\n\n"
            f"Download Video: {video_format}\n"
            f"Download Audio: {audio_format}"
        )
        logger.info("Successfully processed metadata")
        return summary, selected_formats
    except Exception as e:
        logger.error(f"Error processing metadata: {str(e)}")
        return f"Error: {str(e)}", ""

# Build Gradio interface
with gr.Blocks() as ui:
    with gr.Tabs():
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
        
        with gr.TabItem("Split Audio"):
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
            with gr.Row():
                with gr.Column():
                    split_size = gr.Dropdown(
                        label="Split Size",
                        choices=["10s", "30s", "45s", "60s", "100s", "Custom"],
                        type="value",
                        value="60s",
                        interactive=True
                    )
                    custom_split_size = gr.Textbox(
                        label="Custom Split Size (in seconds)",
                        visible=False
                    )
                    overlap_size = gr.Dropdown(
                        label="Overlap",
                        choices=["None", "5s", "10s", "15s", "20s", "25s", "30s", "Custom"],
                        type="value",
                        value="None",
                        interactive=True
                    )
                    custom_overlap_size = gr.Textbox(
                        label="Custom Overlap Size (in seconds)",
                        visible=False
                    )
                    
                    split_size.change(
                        fn=lambda x: update_custom_visibility(x, "split"),
                        inputs=[split_size],
                        outputs=[custom_split_size]
                    )
                    overlap_size.change(
                        fn=lambda x: update_custom_visibility(x, "overlap"),
                        inputs=[overlap_size],
                        outputs=[custom_overlap_size]
                    )
                    split_btn = gr.Button("Split Audio")
                    split_status = gr.Textbox(label="Split Status", interactive=False)
                    
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

# Launch the interface
ui.launch()