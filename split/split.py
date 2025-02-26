import os
from pydub import AudioSegment

# Ensure directories exist
AUDIO_DIR = "audio"
AUDIO_SPLIT_DIR = "audio_split"

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [AUDIO_DIR, AUDIO_SPLIT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

def get_audio_files():
    """Get list of audio files from the audio directory."""
    ensure_directories()
    return [f for f in os.listdir("audio") if f.endswith((".mp3", ".wav", ".m4a"))]

def clean_audio_split_directory():
    """Remove all files from the audio_split directory."""
    if os.path.exists(AUDIO_SPLIT_DIR):
        for file in os.listdir(AUDIO_SPLIT_DIR):
            file_path = os.path.join(AUDIO_SPLIT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def split_audio(file_path, split_size):
    """Split an audio file into chunks of specified size.
    
    Args:
        file_path (str): Path to the audio file to split
        split_size (int): Size of each chunk in seconds
    
    Returns:
        list: List of paths to the split audio files
    """
    ensure_directories()
    
    # Clean up existing files
    clean_audio_split_directory()
    
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Calculate duration and number of chunks
    duration = len(audio)
    chunk_length = split_size * 1000  # Convert to milliseconds
    
    # List to store paths of split files
    split_files = []
    
    # Split the audio into chunks
    for i, start in enumerate(range(0, duration, chunk_length)):
        # Calculate end time for chunk
        end = min(start + chunk_length, duration)
        
        # Extract chunk
        chunk = audio[start:end]
        
        # Generate output filename with simplified format
        output_path = os.path.join(AUDIO_SPLIT_DIR, f"chunk_{i+1:02d}.wav")
        
        # Export chunk
        chunk.export(output_path, format="wav")
        split_files.append(output_path)
    
    return split_files