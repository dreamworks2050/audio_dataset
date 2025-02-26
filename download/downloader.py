import yt_dlp
import os

# Fetch YouTube metadata using yt_dlp
def get_youtube_metadata(url):
    """Fetch metadata for a YouTube video URL."""
    ydl_opts = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info

# Download video based on selected resolution
def download_video(url):
    """Download video in the highest quality format."""
    try:
        metadata = get_youtube_metadata(url)
        video_formats = [
            fmt for fmt in metadata.get('formats', [])
            if fmt.get('vcodec', '').lower() != 'none' and 
               isinstance(fmt.get('height'), int) and
               isinstance(fmt.get('width'), int)
        ]
        
        # Sort video formats by height (resolution) and select the highest quality
        video_formats.sort(key=lambda x: (x.get('height', 0), x.get('width', 0)), reverse=True)
        selected_format = video_formats[0] if video_formats else None
        
        if not selected_format:
            return "No suitable video format found"
            
        ydl_opts = {
            'format': f'{selected_format["format_id"]}+bestaudio/best',
            'outtmpl': f'video/%(title)s_%(height)sp.%(ext)s',
        }
        os.makedirs("video", exist_ok=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "Video download complete"
    except Exception as e:
        return f"Error: {str(e)}"

# Download audio based on selected format
def download_audio(url):
    """Download audio in the highest quality format."""
    try:
        metadata = get_youtube_metadata(url)
        audio_formats = [
            fmt for fmt in metadata.get('formats', [])
            if fmt.get('vcodec', '').lower() == 'none' and 
               fmt.get('acodec', '').lower() != 'none' and 
               isinstance(fmt.get('abr'), (int, float))
        ]
        
        # Sort audio formats by bitrate and select the highest quality
        audio_formats.sort(key=lambda x: float(x.get('abr', 0)), reverse=True)
        selected_format = audio_formats[0] if audio_formats else None
        
        if not selected_format:
            return "No suitable audio format found"
            
        ydl_opts = {
            'format': selected_format['format_id'],
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'outtmpl': 'audio/%(title)s_%(abr)skbps.%(ext)s',
        }
        os.makedirs("audio", exist_ok=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "Audio download complete"
    except Exception as e:
        return f"Error: {str(e)}"