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

## New Feature: AI Analysis

The AI Analysis feature uses Ollama with the "mistral" model to analyze transcription quality. It focuses specifically on the joins between chunks to help identify the best chunking strategy for your transcriptions.

### How It Works

1. The system analyzes each transcription chunk in context with its neighboring chunks
2. It evaluates 8 key criteria for transcription quality:
   - Overall Accuracy
   - Joint Smoothness
   - Contextual Continuity
   - Grammar Integrity
   - Word Completeness
   - Redundancy 
   - Content Loss
   - Joint Readability

3. Each criterion is scored on a scale of 0-10
4. Detailed analysis is provided for each chunk showing the reasoning process
5. Summary reports are generated for each combination of chunk/overlap settings

### Setup Requirements

1. Install Ollama: Follow the instructions at [ollama.com](https://ollama.com)
2. Start the Ollama service (important!):
   - On macOS/Linux: Run `ollama serve` in a terminal
   - On Windows: Ensure the Ollama application is running
3. Pull the models you want to use:
   - For Mistral: `ollama pull mistral`
   - For Llama3: `ollama pull llama3`
   - For other models: `ollama pull model_name`
4. Install Python dependencies: `pip install -r requirements.txt`

### Using the AI Analysis Feature

1. First prepare audio files using the "Prepare Audios" button
2. Transcribe the chunks using the "Transcribe Chunks" button
3. **Make sure Ollama is running** (check with `ps aux | grep ollama`)
4. Navigate to the "AI Analysis" tab
5. Select the Ollama model to use from the dropdown menu (options include "mistral", "llama3", "gemma2:9b", etc.)
6. Click the "Start Analysis" button to begin the analysis process
7. Monitor progress in the "Analysis Status" section
8. View detailed results in the "Analysis Results" section

> **Note:** The analysis process requires that each combination has been transcribed first. Combinations that haven't been transcribed will be skipped during analysis. If you see "Skipped combinations" in the analysis summary, make sure to run the transcription process for those combinations before analyzing them.

### Enhanced Features

The AI Analysis system includes several robust features:

1. **Comprehensive Logging**:
   - Detailed logs for each analysis run
   - Per-file analysis logs with timestamps
   - Summary reports with statistics and recommendations

2. **Error Handling**:
   - Automatic retry for Ollama API failures
   - Graceful handling of missing files
   - Recovery from incomplete transcriptions
   - Detailed error reporting with error types

3. **Detailed Analysis Reports**:
   - Summary files in both JSON and human-readable formats
   - Individual chunk analysis with scores and reasoning
   - Overall statistics and quality metrics
   - Charts and visualizations of results

4. **Metadata Management**:
   - Automatic creation of metadata.json if missing
   - Recovery from combined_transcription.json
   - Validation of transcription files

## Understanding Analysis Results

Each chunk analysis provides:
- Score for each criterion (0-10)
- Step-by-step reasoning
- Overall average score
- Specific observations about the joint quality
- Analysis duration and performance metrics

The system generates several output files:
- `detailed_analysis.log`: Complete log of the analysis process
- `summary.json`: Machine-readable summary of all results
- `summary.txt`: Human-readable summary with recommendations
- Individual chunk analysis files in both JSON and TXT formats

### Analysis Metrics Explained

- **Overall Accuracy**: How accurately the transcription captures the spoken content
- **Joint Smoothness**: How well the chunks connect at their boundaries
- **Contextual Continuity**: Whether the meaning flows naturally across chunk boundaries
- **Grammar Integrity**: Grammatical correctness at chunk boundaries
- **Word Completeness**: Whether words are complete at chunk boundaries
- **Redundancy**: Repeated content across chunk boundaries
- **Content Loss**: Missing content at chunk boundaries
- **Joint Readability**: How readable the text is across chunk boundaries

### Recommendations

The system provides automatic recommendations based on analysis results:
- Suggestions for optimal chunk size and overlap settings
- Identification of problematic combinations
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