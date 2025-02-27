# Audio Dataset Tool Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Detailed Implementation](#detailed-implementation)
5. [Usage Examples](#usage-examples)

## Project Overview

This tool is a comprehensive audio dataset management system that provides functionality for:
- Downloading YouTube videos and extracting audio
- Managing metadata from YouTube videos
- Splitting audio files into smaller segments
- Processing audio with customizable parameters

## Project Structure

```
/audio_dataset/
├── audio/          # Storage for downloaded audio files
├── audio_split/    # Storage for split audio segments
├── download/       # YouTube download functionality
├── logs/           # Application logs
├── metadata/       # YouTube video metadata storage
├── split/          # Audio splitting functionality
├── utils/          # Utility functions and logging
├── video/          # Storage for downloaded videos
├── main.py         # Main application entry point
└── requirements.txt # Project dependencies
```

## Core Components

### 1. Main Application (main.py)

#### Key Functions

##### `save_metadata(metadata)`
- **Purpose**: Saves YouTube video metadata to a JSON file
- **Parameters**:
  - `metadata` (dict): YouTube video metadata
- **Returns**: Filename where metadata was saved
- **Example**:
```python
metadata = get_youtube_metadata("https://youtube.com/watch?v=example")
filename = save_metadata(metadata)
# Saves to: metadata/metadata_01.txt
```

##### `summarize_metadata(metadata)`
- **Purpose**: Processes and formats video metadata for display
- **Parameters**:
  - `metadata` (dict): YouTube video metadata
- **Returns**: Tuple of (summary, video_format, audio_format)
- **Implementation Details**:
  1. Extracts basic information:
     - Title
     - Duration (converted to MM:SS format)
     - Upload date (converted from YYYYMMDD to readable format)
  2. Processes video formats:
     - Filters for valid video formats
     - Sorts by quality (resolution)
     - Selects highest quality format
  3. Processes audio formats:
     - Filters for audio-only formats
     - Sorts by bitrate
     - Selects highest quality format
- **Example Usage**:
```python
metadata = get_youtube_metadata(url)
summary, video_fmt, audio_fmt = summarize_metadata(metadata)
# Returns formatted strings for display
```

##### `grab_metadata(url)`
- **Purpose**: Fetches and processes YouTube video metadata
- **Parameters**:
  - `url` (str): YouTube video URL
- **Returns**: Tuple of (summary, selected_formats)
- **Error Handling**: Returns error message if URL is invalid or fetch fails

### 2. Download Module (download/downloader.py)

#### Key Functions

##### `get_youtube_metadata(url)`
- **Purpose**: Fetches metadata from YouTube videos
- **Parameters**:
  - `url` (str): YouTube video URL
- **Returns**: Dictionary containing video metadata
- **Used By**: `grab_metadata()`, `download_video()`, `download_audio()`

##### `download_video(url)`
- **Purpose**: Downloads video in highest quality format
- **Parameters**:
  - `url` (str): YouTube video URL
- **Returns**: Status message
- **Implementation Details**:
  1. Fetches metadata
  2. Filters for video formats
  3. Selects highest quality format
  4. Downloads using yt-dlp

##### `download_audio(url)`
- **Purpose**: Downloads audio in highest quality format
- **Parameters**:
  - `url` (str): YouTube video URL
- **Returns**: Status message

### 3. Split Module (split/split.py)

#### Key Functions

##### `get_audio_files()`
- **Purpose**: Lists available audio files for processing
- **Returns**: List of audio file names
- **Used By**: Gradio UI dropdown

##### `split_audio(file_path, split_size, overlap_size)`
- **Purpose**: Splits audio files into segments
- **Parameters**:
  - `file_path` (str): Path to audio file
  - `split_size` (int): Segment duration in seconds
  - `overlap_size` (int): Overlap duration in seconds
- **Returns**: Tuple of (split_files, summary)

### 4. User Interface (Gradio Components)

#### Download Tab
- **Components**:
  - URL input textbox
  - Metadata display
  - Format display
  - Status display
  - Action buttons:
    - Grab Information
    - Download Video
    - Download Audio

#### Split Audio Tab
- **Components**:
  - Audio file upload/selection
  - Split size dropdown
  - Custom split size input
  - Overlap size dropdown
  - Custom overlap size input
  - Split button
  - Status display

#### Event Handlers
- **URL Input**: Triggers metadata fetch on button click
- **Download Buttons**: Trigger respective download functions
- **Split Button**: Processes audio splitting with selected parameters
- **Dropdown Changes**: Update UI visibility based on selections

## Usage Examples

### 1. Downloading a Video
```python
# 1. Enter YouTube URL
url = "https://youtube.com/watch?v=example"

# 2. Fetch metadata
summary, formats = grab_metadata(url)

# 3. Download video
status = download_video(url)
```

### 2. Splitting Audio
```python
# 1. Select audio file
audio_path = "audio/example.mp3"

# 2. Set parameters
split_size = 30  # 30 seconds
overlap = 5      # 5 seconds overlap

# 3. Process split
split_files, summary = split_audio(audio_path, split_size, overlap)
```

## Dependencies

- gradio: Web interface framework
- yt-dlp: YouTube download functionality
- pydub: Audio processing
- datetime: Date formatting
- json: Metadata storage
- os: File system operations

## Error Handling

The application implements comprehensive error handling:
1. Invalid URLs
2. Network errors
3. File system errors
4. Invalid input parameters
5. Processing errors

All errors are logged and user-friendly messages are displayed in the UI.

## Best Practices

1. **Metadata Management**:
   - Stored in structured JSON format
   - Includes comprehensive video information
   - Cached for efficiency

2. **Audio Processing**:
   - Supports multiple formats
   - Configurable parameters
   - Progress tracking

3. **User Interface**:
   - Intuitive layout
   - Real-time feedback
   - Error messaging
   - Progress updates

4. **File Management**:
   - Organized directory structure
   - Consistent naming conventions
   - Automatic directory creation

## Logging

Logging is implemented through the utils/logger.py module:
- Error logging
- Operation tracking
- Debug information
- Performance metrics

## Future Enhancements

1. Batch processing
2. Additional format support
3. Advanced audio processing
4. Enhanced metadata analysis
5. Progress tracking improvements

---

This documentation provides a comprehensive overview of the audio dataset tool's implementation, usage, and maintenance. For specific questions or issues, please refer to the relevant sections or consult the source code comments.