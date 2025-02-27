# YouTube Video Downloader v0.1.2

A Python-based tool with a user-friendly interface for downloading YouTube videos, extracting high-quality audio tracks, and splitting audio files into smaller chunks with advanced overlap control.

## Features

- Easy-to-use interface for downloading videos and audio from YouTube
- Real-time metadata display showing video information before download
- Multiple format options for both video and audio downloads
- Advanced audio splitting functionality with:
  - Customizable chunk sizes
  - Configurable overlap settings
  - Detailed overlap validation
  - Real-time splitting progress
- Clean and organized file management
- Audio transcription capabilities using whisper.cpp

## Project Structure

```
./
├── audio/           # Directory for extracted audio files
├── audio_split/     # Directory for split audio chunks
├── audio_text/      # Directory for transcription outputs
├── download/        # Core downloading functionality
├── logs/            # Application logs
├── metadata/        # Stored video metadata files
├── split/           # Audio splitting functionality
├── transcribe/      # Audio transcription functionality
├── utils/           # Utility functions
├── video/           # Directory for downloaded videos
├── whisper.cpp/     # Whisper.cpp library for transcription
├── main.py          # Main application entry point
├── requirements.txt # Project dependencies
└── metadata_schema.md  # Detailed metadata schema documentation
```

## Function Reference

### Main Application Functions

#### Metadata Management
- `save_metadata(metadata)`: Saves YouTube video metadata to a JSON file in the 'metadata' directory.
- `summarize_metadata(metadata)`: Processes metadata and extracts relevant information for display, including video title, duration, upload date, and available formats.
- `grab_metadata(url)`: Fetches metadata for a YouTube URL and updates the Gradio UI components.

### Download Module Functions

- `get_youtube_metadata(url)`: Fetches comprehensive metadata for a YouTube video URL using yt_dlp.
- `download_video(url)`: Downloads a YouTube video in the highest quality format available, automatically selecting the best resolution.
- `download_audio(url)`: Downloads and extracts high-quality audio from a YouTube video, converting it to WAV format with 16kHz sample rate and mono channel.

### Audio Splitting Module Functions

- `ensure_directories()`: Creates necessary directories for audio processing if they don't exist.
- `get_audio_files()`: Returns a list of audio files from the audio directory.
- `clean_audio_split_directory(preserve_state=True)`: Removes all files from the audio_split directory, with an option to preserve the split state file.
- `validate_audio_file(file_path)`: Validates and locates an audio file, checking both the provided path and the audio directory.
- `validate_overlap_calculations(chunk_length_ms, overlap_ms, duration, total_chunks)`: Validates overlap calculations before processing chunks to ensure consistency.
- `split_audio(file_path, split_size, overlap_size=0)`: Splits an audio file into chunks of specified size with optional overlap.
- `verify_split_results()`: Verifies the results of the audio splitting process.
- `fix_split_overlaps()`: Fixes any issues with overlaps between audio chunks.
- `test_overlap_calculations(duration_seconds, chunk_sizes, overlap_sizes)`: Tests overlap calculations with various parameters.
- `run_overlap_test(duration, chunk_size, overlap_size)`: Runs a specific overlap test with given parameters.

### Transcription Module Functions

#### TranscriptionService Class
- `__init__(model_path)`: Initializes the transcription service with a specified whisper.cpp model.
- `_verify_dependencies()`: Verifies that the required dependencies (whisper.cpp binary and model) exist.
- `transcribe_chunk(audio_chunk, lang_code="en", initial_prompt=None)`: Transcribes an audio chunk using whisper.cpp with optional language code and initial prompt.

### Utility Functions

#### Logger Class
- `__init__(name='audio_dataset')`: Initializes the logger with console and file handlers.
- `clear_logs()`: Clears the log file by truncating it to zero size.
- `debug(message)`, `info(message)`, `warning(message)`, `error(message)`, `critical(message)`: Log messages at different severity levels.

#### Cleanup Functions
- `cleanup_python_cache()`: Removes all Python cache files and directories to keep the codebase clean.

## User Interface Guide

### Download Tab

#### YouTube Video/Audio Download
1. **URL Input**
   - Paste a YouTube URL into the text box
   - Click "Grab Information" to fetch video details

2. **Metadata Display**
   - View comprehensive video information including:
     - Title
     - Duration
     - Upload date
     - Available video and audio formats

3. **Download Options**
   - "Download Video": Downloads the highest quality video format
   - "Download Audio": Extracts high-quality audio track

### Split Audio Tab

#### Audio Processing
1. **Input Options**
   - Upload a new audio file using the audio upload component
   - Select an existing audio file from the dropdown menu

2. **Split Settings**
   - Choose from preset split durations (10s, 30s, 45s, 60s, 100s)
   - Select "Custom" for a specific duration in seconds
   - Set overlap duration between chunks:
     - Choose from preset overlap durations
     - Enter custom overlap duration
     - Disable overlap with "None" option

3. **Processing**
   - Click "Split Audio" to divide the file into chunks
   - View detailed processing information:
     - Chunk timecodes and durations
     - Overlap validation results
     - Processing summary

### Transcription Tab

#### Audio Transcription
1. **Model Selection**
   - Choose from available whisper.cpp models
   - Models vary in size and accuracy

2. **Input Selection**
   - Select audio files to transcribe from the audio_split directory
   - Batch processing of multiple files is supported

3. **Transcription Options**
   - Set language code for transcription (default: English)
   - Provide optional initial prompt to guide transcription
   - Configure processing parameters

4. **Output**
   - View transcription results in real-time
   - Results are saved to the audio_text directory

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up whisper.cpp:
   ```bash
   python setup_whisper.py
   ```
4. Run the application:
   ```bash
   python main.py
   ```

## Version History

### v0.1.2 (Current)

- Added customizable overlap settings for audio splitting
- Implemented detailed overlap validation
- Enhanced progress reporting with chunk information
- Improved error handling and validation
- Added transcription capabilities with whisper.cpp integration

### v0.1.1

- Initial release with user-friendly interface
- Support for high-quality video and audio downloads
- Implementation of audio splitting feature
- Real-time metadata display
- Organized file management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.