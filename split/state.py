import os
import json
from datetime import datetime
from utils.logger import logger

CHUNKS_DIR = "audio_split"
STATE_FILE = os.path.join(CHUNKS_DIR, "split_state.json")

def save_split_state(original_file, split_size, overlap_size, split_files, total_duration=None):
    """Save the state of an audio splitting operation to a JSON file.
    
    Args:
        original_file (str): Name of the original audio file
        split_size (int): Size of each chunk in seconds
        overlap_size (int): Size of overlap between chunks in seconds
        split_files (list): List of paths to the split audio files
        total_duration (float, optional): Total duration of the audio in seconds
    """
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    
    state = {
        "original_file": original_file,
        "split_timestamp": datetime.now().isoformat(),
        "settings": {
            "chunk_length": split_size,
            "overlap_size": overlap_size
        },
        "chunks": {
            "total_count": len(split_files),
            "files": [os.path.basename(f) for f in split_files]
        },
        "total_duration": total_duration
    }
    
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Split state saved to {STATE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save split state: {str(e)}")

def get_split_state():
    """Get the current state of audio splits.
    
    Returns:
        dict: The current split state or None if no state file exists
    """
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read split state: {str(e)}")
    return None

def get_split_summary():
    """Generate a human-readable summary of the current split state.
    
    Returns:
        str: A formatted summary of the split state
    """
    state = get_split_state()
    if not state:
        return "No audio splits found."
    
    # Format timestamp
    timestamp = datetime.fromisoformat(state['split_timestamp'])
    formatted_time = timestamp.strftime("%B %d, %Y at %I:%M %p")
    
    # Format duration
    total_duration = state.get('total_duration')
    if total_duration:
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        duration_str = "Unknown"
    
    summary = (
        f"Last Split Operation:\n\n"
        f"Original File: {state['original_file']}\n"
        f"Split on: {formatted_time}\n\n"
        f"Settings:\n"
        f"- Chunk Length: {state['settings']['chunk_length']} seconds\n"
        f"- Overlap Size: {state['settings']['overlap_size']} seconds\n\n"
        f"Results:\n"
        f"- Total Duration: {duration_str}\n"
        f"- Total Chunks: {state['chunks']['total_count']}\n"
    )
    
    return summary