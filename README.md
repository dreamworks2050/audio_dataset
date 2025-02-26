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

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Version History

### v0.1.2 (Current)

- Added customizable overlap settings for audio splitting
- Implemented detailed overlap validation
- Enhanced progress reporting with chunk information
- Improved error handling and validation

### v0.1.1

- Initial release with user-friendly interface
- Support for high-quality video and audio downloads
- Implementation of audio splitting feature
- Real-time metadata display
- Organized file management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.