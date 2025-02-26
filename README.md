# YouTube Video Downloader v0.1.1

A Python-based tool for downloading YouTube videos and extracting high-quality audio tracks with comprehensive metadata handling.

## Features

- Download YouTube videos in highest available quality
- Extract high-quality audio from YouTube videos
- Automatic metadata extraction and storage
- Organized file structure for videos and audio files
- Support for multiple video formats and resolutions
- Intelligent format selection for best quality

## Project Structure

```
./
├── audio/           # Directory for extracted audio files
├── download/        # Core downloading functionality
├── metadata/        # Stored video metadata files
├── video/          # Directory for downloaded videos
├── main.py         # Main application entry point
└── metadata_schema.md  # Detailed metadata schema documentation
```

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Required dependencies:
- yt-dlp (>=2023.12.30)
- requests (>=2.31.0)
- datetime (>=5.4)
- python-dateutil (>=2.8.2)
- tqdm (>=4.66.1)

## Usage

### Basic Usage

Run the main script with a YouTube URL:

```bash
python main.py [YouTube_URL]
```

### Output Structure

- Downloaded videos are stored in the `video/` directory
- Extracted audio files are stored in the `audio/` directory
- Video metadata is saved in the `metadata/` directory

## Quality Selection

The downloader automatically selects the highest quality formats available:

### Video Selection
- Searches for formats with video codec present (`vcodec != "none"`)
- Requires both height and width specifications
- Sorts by resolution (height × width) to select the best quality
- Prioritizes formats with better video codecs when available

### Audio Selection
- Identifies audio-only formats (`vcodec == "none" && acodec != "none"`)
- Requires audio bitrate (abr) information
- Sorts by audio bitrate to select the highest quality
- Typically selects formats with the best audio codec available

## Metadata Handling

The project implements comprehensive metadata handling:

- Extracts and stores detailed format information
- Maintains technical specifications (codecs, bitrates)
- Preserves descriptive information (format notes, resolution)
- Supports both video-only and audio-only format tracking

## Development

### Core Components

1. **Downloader Module** (`download/downloader.py`)
   - Handles video and audio downloads
   - Implements intelligent format selection
   - Manages file organization

2. **Main Script** (`main.py`)
   - Processes command line arguments
   - Coordinates download operations
   - Manages metadata extraction

## Version History

### v0.1.1
- Initial release with core functionality
- Support for high-quality video and audio downloads
- Comprehensive metadata handling
- Organized file structure implementation
- Intelligent format selection system