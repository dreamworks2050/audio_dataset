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
            logger.debug(f"Using yt_dlp to extract info without downloading")
            info = ydl.extract_info(url, download=False)
            logger.debug(f"Successfully fetched metadata for video: {info.get('title', 'Unknown')}")
            return info
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {str(e)}")
        return None

# Download video based on selected resolution
def download_video(url):
    """Download video in the highest quality format."""
    logger.info(f"Starting video download process for URL: {url}")
    try:
        logger.debug("Attempting to fetch video metadata")
        metadata = get_youtube_metadata(url)
        if not metadata:
            logger.error("Failed to fetch video metadata, aborting download")
            return "Failed to fetch video metadata"
            
        logger.debug(f"Successfully retrieved metadata for video: {metadata.get('title', 'Unknown')}")
        logger.debug(f"Video duration: {metadata.get('duration_string', 'Unknown')}, Upload date: {metadata.get('upload_date', 'Unknown')}")
        
        logger.debug("Filtering available formats to find suitable video formats")
        video_formats = [
            fmt for fmt in metadata.get('formats', [])
            if fmt.get('vcodec', '').lower() != 'none' and 
               isinstance(fmt.get('height'), int) and
               isinstance(fmt.get('width'), int)
        ]
        
        logger.debug(f"Found {len(video_formats)} suitable video formats")
        
        # Sort video formats by height (resolution) and select the highest quality
        logger.debug("Sorting video formats by resolution to select highest quality")
        video_formats.sort(key=lambda x: (x.get('height', 0), x.get('width', 0)), reverse=True)
        selected_format = video_formats[0] if video_formats else None
        
        if not selected_format:
            logger.warning("No suitable video format found, aborting download")
            return "No suitable video format found"
            
        logger.info(f"Selected video format: {selected_format.get('height')}p, Format ID: {selected_format.get('format_id')}")
        logger.debug(f"Selected format details - Codec: {selected_format.get('vcodec')}, FPS: {selected_format.get('fps')}, File size: {selected_format.get('filesize_approx', 'Unknown')} bytes")
            
        ydl_opts = {
            'format': f'{selected_format["format_id"]}+bestaudio/best',
            'outtmpl': f'video/%(title)s_%(height)sp.%(ext)s',
        }
        
        logger.debug(f"Creating video directory if it doesn't exist")
        os.makedirs("video", exist_ok=True)
        
        logger.info(f"Starting video download with options: {ydl_opts}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.debug("Initialized YoutubeDL with configured options")
            ydl.download([url])
            
        # Log the files in the video directory for debugging
        video_files = os.listdir("video")
        logger.debug(f"Files in video directory after download: {video_files}")
        
        logger.info(f"Video download completed successfully for: {metadata.get('title', 'Unknown')}")
        return "Video download complete"
    except Exception as e:
        error_msg = f"Error downloading video: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Exception details: {type(e).__name__}")
        return f"Error: {str(e)}"

# Download audio based on selected format
def download_audio(url):
    """Download audio in the highest quality format."""
    logger.info(f"Starting audio download process for URL: {url}")
    try:
        logger.debug("Attempting to fetch audio metadata")
        metadata = get_youtube_metadata(url)
        if not metadata:
            logger.error("Failed to fetch audio metadata, aborting download")
            return "Failed to fetch audio metadata"
            
        logger.debug(f"Successfully retrieved metadata for audio: {metadata.get('title', 'Unknown')}")
        logger.debug(f"Audio duration: {metadata.get('duration_string', 'Unknown')}, Upload date: {metadata.get('upload_date', 'Unknown')}")
        
        logger.debug("Filtering available formats to find suitable audio formats")
        audio_formats = [
            fmt for fmt in metadata.get('formats', [])
            if fmt.get('vcodec', '').lower() == 'none' and 
               fmt.get('acodec', '').lower() != 'none' and 
               isinstance(fmt.get('abr'), (int, float))
        ]
        
        logger.debug(f"Found {len(audio_formats)} suitable audio formats")
        
        # Sort audio formats by bitrate and select the highest quality
        logger.debug("Sorting audio formats by bitrate to select highest quality")
        audio_formats.sort(key=lambda x: float(x.get('abr', 0)), reverse=True)
        selected_format = audio_formats[0] if audio_formats else None
        
        if not selected_format:
            logger.warning("No suitable audio format found, aborting download")
            return "No suitable audio format found"
            
        logger.info(f"Selected audio format: {selected_format.get('abr')}kbps, Format ID: {selected_format.get('format_id')}")
        logger.debug(f"Selected format details - Codec: {selected_format.get('acodec')}, Channels: {selected_format.get('audio_channels')}, File size: {selected_format.get('filesize_approx', 'Unknown')} bytes")
            
        ydl_opts = {
            'format': selected_format['format_id'],
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            # Use direct FFmpeg options without the postprocessor prefix
            'postprocessor_args': {
                'FFmpegExtractAudio': ['-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le'],
            },
            'outtmpl': 'audio/%(title)s_%(abr)skbps.%(ext)s',
            'verbose': True
        }
        
        logger.debug(f"Creating audio directory if it doesn't exist")
        os.makedirs("audio", exist_ok=True)
        
        logger.info(f"Starting audio download with options: {ydl_opts}")
        logger.debug(f"Audio will be converted to WAV format with 16kHz sample rate and mono channel")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.debug("Initialized YoutubeDL with configured options")
            ydl.download([url])
        
        # Log the files in the audio directory for debugging
        audio_files = os.listdir("audio")
        logger.info(f"Files in audio directory after download: {audio_files}")
        
        if not audio_files:
            logger.warning("No audio files found in the output directory after download")
        else:
            logger.debug(f"Successfully downloaded {len(audio_files)} audio file(s)")
        
        logger.info(f"Audio download completed successfully for: {metadata.get('title', 'Unknown')}")
        return "Audio download complete"
    except Exception as e:
        error_msg = f"Error downloading audio: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Exception details: {type(e).__name__}")
        return f"Error: {str(e)}"