# YouTube Video Downloader v0.1.1

A Python-based tool for downloading YouTube videos and extracting high-quality audio tracks with comprehensive metadata handling.

## Features

- High-quality video and audio downloads from YouTube
- Intelligent format selection for optimal quality
- Comprehensive metadata extraction and storage
- Audio file splitting functionality with customizable chunk sizes
- Clean and organized file management
- Progress tracking with tqdm integration

## Project Structure

```
./
├── audio/           # Directory for extracted audio files
├── audio_split/     # Directory for split audio chunks
├── download/        # Core downloading functionality
├── metadata/        # Stored video metadata files
├── video/          # Directory for downloaded videos
├── split/          # Audio splitting functionality
├── main.py         # Main application entry point
├── requirements.txt # Project dependencies
└── metadata_schema.md  # Detailed metadata schema documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/youtube-video-downloader.git
   cd youtube-video-downloader
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies

- yt-dlp (>=2023.12.30) - Core YouTube download functionality
- requests (>=2.31.0) - HTTP requests handling
- datetime (>=5.4) - Date and time operations
- python-dateutil (>=2.8.2) - Advanced date parsing
- tqdm (>=4.66.1) - Progress bar functionality

## Usage

### Basic Usage

1. Download a video:
   ```python
   from download.downloader import download_video
   
   url = "https://www.youtube.com/watch?v=example"
   status = download_video(url)
   print(status)
   ```

2. Download audio only:
   ```python
   from download.downloader import download_audio
   
   url = "https://www.youtube.com/watch?v=example"
   status = download_audio(url)
   print(status)
   ```

3. Split audio into chunks:
   ```python
   from split.split import split_audio
   
   file_path = "audio/example.mp3"
   split_size = 300  # 5 minutes in seconds
   split_files = split_audio(file_path, split_size)
   print(f"Created {len(split_files)} chunks")
   ```

## Function Documentation

### Main Module (main.py)

#### save_metadata(metadata)
Saves video metadata to a JSON file in the 'metadata' directory.
- **Parameters:**
  - metadata (dict): The video metadata to save
- **Returns:** str - Path to the saved metadata file

#### summarize_metadata(metadata)
Processes metadata and extracts relevant information for display and downloads.
- **Parameters:**
  - metadata (dict): Raw video metadata
- **Returns:** tuple (str, str, str) - Summary text, video format, and audio format

#### grab_metadata(url)
Fetches metadata and updates Gradio components.
- **Parameters:**
  - url (str): YouTube video URL
- **Returns:** tuple (str, str) - Metadata summary and selected formats

### Downloader Module (download/downloader.py)

#### get_youtube_metadata(url)
Fetches metadata for a YouTube video URL.
- **Parameters:**
  - url (str): YouTube video URL
- **Returns:** dict - Video metadata

#### download_video(url)
Downloads video in the highest quality format.
- **Parameters:**
  - url (str): YouTube video URL
- **Returns:** str - Status message

#### download_audio(url)
Downloads audio in the highest quality format.
- **Parameters:**
  - url (str): YouTube video URL
- **Returns:** str - Status message

### Split Module (split/split.py)

#### ensure_directories()
Creates necessary directories if they don't exist.
- **Parameters:** None
- **Returns:** None

#### get_audio_files()
Gets list of audio files from the audio directory.
- **Parameters:** None
- **Returns:** list - List of audio file names

#### clean_audio_split_directory()
Removes all files from the audio_split directory.
- **Parameters:** None
- **Returns:** None

#### split_audio(file_path, split_size)
Splits an audio file into chunks of specified size.
- **Parameters:**
  - file_path (str): Path to the audio file to split
  - split_size (int): Size of each chunk in seconds
- **Returns:** list - List of paths to the split audio files

## Development

### Core Components

1. **Downloader Module** (`download/downloader.py`)
   - Handles video and audio downloads
   - Implements intelligent format selection
   - Manages file organization
   - Supports various quality options
   - Handles download progress tracking

2. **Main Script** (`main.py`)
   - Processes command line arguments
   - Coordinates download operations
   - Manages metadata extraction
   - Provides user interface
   - Handles error reporting

3. **Split Module** (`split/split.py`)
   - Handles audio file splitting
   - Manages split file organization
   - Provides audio processing utilities
   - Ensures clean directory management
   - Supports various audio formats

## Error Handling

The application includes comprehensive error handling for:
- Invalid YouTube URLs
- Network connectivity issues
- File system operations
- Format compatibility
- Resource availability

## Version History

### v0.1.1 (Current)
- Initial release with core functionality
- Support for high-quality video and audio downloads
- Implementation of audio splitting feature
- Basic metadata handling
- Directory management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.