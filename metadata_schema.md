# YouTube Video Metadata Schema

This document describes the structure of the metadata JSON file generated for YouTube videos.

## Root Level Properties

```json
{
  "id": "QGICMZ2eoK0",              // YouTube video ID (string)
  "title": "Video Title",           // Video title (string)
  "formats": [...],                  // Array of format objects
  "duration_string": "7:00:22",     // Duration in HH:MM:SS format
  "upload_date": "20250116",        // Upload date in YYYYMMDD format
  "channel": "Channel Name",         // Channel name
  "channel_follower_count": 64000,  // Number of channel followers
  "like_count": 3670,               // Number of likes
  "is_live": false,                 // Whether the video is currently live
  "was_live": true                  // Whether the video was a live stream
}
```

## Format Object Structure

The `formats` array contains different types of format objects, each serving a specific purpose:

### 1. Video Download Formats
```json
{
  "format_id": "136",              // Format identifier for video quality (e.g., "136" for 720p)
  "format_note": "720p",          // Human-readable quality description
  "filesize": 1982679143,         // Size in bytes
  "fps": 30,                      // Frames per second
  "height": 720,                  // Video height in pixels
  "width": 1280,                  // Video width in pixels
  "vcodec": "avc1.4d401f",        // Video codec used
  "acodec": "none",               // No audio in video-only formats
  "ext": "mp4",                   // File extension
  "vbr": 1500,                    // Video bitrate (Kbps)
  "resolution": "1280x720"         // Resolution string
}
```

### 2. Audio Download Formats
```json
{
  "format_id": "140",              // Format identifier for audio quality
  "format_note": "medium",        // Audio quality description
  "asr": 44100,                   // Audio sample rate (Hz)
  "filesize": 408184193,          // Size in bytes
  "acodec": "mp4a.40.2",          // Audio codec used
  "vcodec": "none",               // No video in audio-only formats
  "abr": 129.471,                 // Audio bitrate (Kbps)
  "ext": "m4a"                     // File extension
}
```

### 3. Storyboard Formats (Thumbnails)
```json
{
  "format_id": "sb3",              // Storyboard format identifier
  "format_note": "storyboard",     // Format type
  "ext": "mhtml",                  // File extension
  "width": 48,                     // Thumbnail width
  "height": 27,                    // Thumbnail height
  "fps": 0.003964,                // Frames per second
  "rows": 10,                     // Number of rows in storyboard
  "columns": 10,                  // Number of columns in storyboard
  "fragments": [                   // Array of storyboard fragments
    {
      "url": "https://...",       // URL to storyboard image
      "duration": 25222.0         // Duration covered by this fragment
    }
  ]
}
```

## Special Format Types

### Video Formats
Video formats have these additional characteristics:
- `vcodec` is not "none"
- `height` and `width` are present
- Common `format_id` values: "136", "243", etc.

Example:
```json
{
  "format_id": "136",
  "height": 720,
  "width": 1280,
  "vcodec": "avc1.64001f",
  "acodec": "none"
}
```

### Audio Formats
Audio formats have these characteristics:
- `acodec` is not "none"
- `vcodec` is "none"
- `abr` (audio bitrate) is present

Example:
```json
{
  "format_id": "140",
  "format_note": "medium",
  "asr": 44100,
  "filesize": 408184193,
  "acodec": "mp4a.40.2",
  "vcodec": "none",
  "abr": 129.471
}
```

## Special Format Types

### Video Formats
Video formats have these additional characteristics:
- `vcodec` is not "none"
- `height` and `width` are present
- Common `format_id` values: "136", "243", etc.

Example:
```json
{
  "format_id": "136",
  "height": 720,
  "width": 1280,
  "vcodec": "avc1.64001f",
  "acodec": "none"
}
```

### Audio Formats
Audio formats have these characteristics:
- `acodec` is not "none"
- `vcodec` is "none"
- `abr` (audio bitrate) is present

Example:
```json
{
  "format_id": "140",
  "format_note": "medium",
  "asr": 44100,
  "filesize": 408184193,
  "acodec": "mp4a.40.2",
  "vcodec": "none",
  "abr": 129.471
}
```

## Notes

1. The metadata file contains comprehensive information about available video and audio formats for a YouTube video.
2. Each format entry represents a different quality or encoding option.
3. The file includes both technical specifications (codecs, bitrates) and descriptive information (format notes, resolution).
4. Some fields may be null or "none" depending on the format type (video-only, audio-only, or combined).
5. For downloading purposes, focus on formats where:
   - For video: `vcodec` is not "none" and `height` is present
   - For audio: `acodec` is not "none" and `abr` is present

## Locating Download Formats in Metadata Structure

To find suitable video and audio formats for downloading, navigate the following structure:

```
root
└── formats (array)
    ├── video formats (objects where vcodec != "none")
    │   └── characteristics:
    │       - format_id (e.g., "136", "243")
    │       - height and width present
    │       - vcodec != "none"
    │       - acodec == "none" (video-only)
    │
    └── audio formats (objects where acodec != "none")
        └── characteristics:
            - format_id (e.g., "140")
            - vcodec == "none"
            - acodec != "none"
            - abr present (audio bitrate)
```

### How to Access Download Formats

1. Start at the root object of the metadata JSON
2. Access the `formats` array
3. Iterate through the formats array to find suitable entries:

#### For Video Downloads:
```json
{
  "formats": [  // Root level array
    {           // Video format entry
      "format_id": "136",
      "vcodec": "avc1.4d401f",  // Must NOT be "none"
      "height": 720,           // Must be present
      "width": 1280,          // Must be present
      "acodec": "none"        // Video-only format
    }
    // ... other formats ...
  ]
}
```

#### For Audio Downloads:
```json
{
  "formats": [  // Root level array
    {           // Audio format entry
      "format_id": "140",
      "vcodec": "none",        // Must be "none"
      "acodec": "mp4a.40.2",   // Must NOT be "none"
      "abr": 129.471          // Audio bitrate must be present
    }
    // ... other formats ...
  ]
}
```

### Selection Criteria

1. **For Video Downloads:**
   - Path: `root.formats[].{format object}`
   - Required conditions:
     ```python
     format["vcodec"] != "none" and
     "height" in format and
     "width" in format
     ```
   - Best practice: Sort by height/resolution for quality selection

2. **For Audio Downloads:**
   - Path: `root.formats[].{format object}`
   - Required conditions:
     ```python
     format["vcodec"] == "none" and
     format["acodec"] != "none" and
     "abr" in format
     ```
   - Best practice: Sort by abr (audio bitrate) for quality selection

### Example Code for Format Selection

```python
# Get video formats
video_formats = [
    f for f in metadata["formats"]
    if f["vcodec"] != "none" and "height" in f and "width" in f
]

# Get audio formats
audio_formats = [
    f for f in metadata["formats"]
    if f["vcodec"] == "none" and f["acodec"] != "none" and "abr" in f
]

# Sort by quality
video_formats.sort(key=lambda x: (x.get("height", 0), x.get("width", 0)), reverse=True)
audio_formats.sort(key=lambda x: x.get("abr", 0), reverse=True)

# Get best quality formats
best_video = video_formats[0] if video_formats else None
best_audio = audio_formats[0] if audio_formats else None
```