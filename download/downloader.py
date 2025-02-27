import yt_dlp
import os
from utils.logger import logger

# Fetch YouTube metadata using yt_dlp
def get_youtube_metadata(url):
    """Fetch metadata for a YouTube video URL."""
    logger.info(f"Fetching metadata for URL: {url}")
    ydl_opts = {'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            logger.debug(f"Successfully fetched metadata for video: {info.get('title', 'Unknown')}")
            return info
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {str(e)}")
        return None

# Download video based on selected resolution
def download_video(url):
    """Download video in the highest quality format."""
    try:
        metadata = get_youtube_metadata(url)
        if not metadata:
            return "Failed to fetch video metadata"
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
            logger.warning("No suitable video format found")
            return "No suitable video format found"
            
        logger.info(f"Selected video format: {selected_format.get('height')}p")
            
        ydl_opts = {
            'format': f'{selected_format["format_id"]}+bestaudio/best',
            'outtmpl': f'video/%(title)s_%(height)sp.%(ext)s',
        }
        os.makedirs("video", exist_ok=True)
        logger.debug("Starting video download")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info("Video download completed successfully")
        return "Video download complete"
    except Exception as e:
        error_msg = f"Error downloading video: {str(e)}"
        logger.error(error_msg)
        return f"Error: {str(e)}"

# Download audio based on selected format
def download_audio(url):
    """Download audio in the highest quality format."""
    try:
        metadata = get_youtube_metadata(url)
        if not metadata:
            return "Failed to fetch audio metadata"
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
            logger.warning("No suitable audio format found")
            return "No suitable audio format found"
            
        logger.info(f"Selected audio format: {selected_format.get('abr')}kbps")
            
        ydl_opts = {
            'format': selected_format['format_id'],
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'outtmpl': 'audio/%(title)s_%(abr)skbps.%(ext)s',
        }
        os.makedirs("audio", exist_ok=True)
        logger.debug("Starting audio download")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info("Audio download completed successfully")
        return "Audio download complete"
    except Exception as e:
        error_msg = f"Error downloading audio: {str(e)}"
        logger.error(error_msg)
        return f"Error: {str(e)}"