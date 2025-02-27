import os
from pydub import AudioSegment
from utils.logger import logger
from .state import save_split_state

# Ensure directories exist
AUDIO_DIR = "audio"
AUDIO_SPLIT_DIR = "audio_split"

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [AUDIO_DIR, AUDIO_SPLIT_DIR]:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory)
        else:
            logger.debug(f"Directory already exists: {directory}")

def get_audio_files():
    """Get list of audio files from the audio directory."""
    ensure_directories()
    files = [f for f in os.listdir("audio") if f.endswith((".mp3", ".wav", ".m4a"))]
    logger.info(f"Found {len(files)} audio files in {AUDIO_DIR}")
    return files

def clean_audio_split_directory():
    """Remove all files from the audio_split directory."""
    if os.path.exists(AUDIO_SPLIT_DIR):
        logger.info(f"Cleaning {AUDIO_SPLIT_DIR} directory")
        count = 0
        for file in os.listdir(AUDIO_SPLIT_DIR):
            file_path = os.path.join(AUDIO_SPLIT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                count += 1
        logger.debug(f"Removed {count} files from {AUDIO_SPLIT_DIR}")

def split_audio(file_path, split_size, overlap_size=0):
    """Split an audio file into chunks of specified size with optional overlap.
    
    Args:
        file_path (str): Path to the audio file to split
        split_size (int): Size of each chunk in seconds
        overlap_size (int, optional): Size of overlap between chunks in seconds. Defaults to 0.
    
    Returns:
        tuple: (List of paths to the split audio files, Summary string)
    """
    ensure_directories()
    
    # Clean up existing files
    clean_audio_split_directory()
    
    logger.info(f"Loading audio file: {os.path.basename(file_path)}")
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        logger.debug(f"Successfully loaded audio file: {os.path.basename(file_path)}")
    except Exception as e:
        logger.error(f"Failed to load audio file {os.path.basename(file_path)}: {str(e)}")
        return [], "Error: Failed to load audio file"
    
    # Calculate duration and number of chunks
    duration = len(audio)
    chunk_length = split_size * 1000  # Convert to milliseconds
    overlap_ms = overlap_size * 1000
    total_chunks = (duration + chunk_length - 1) // chunk_length
    
    logger.info(f"Audio duration: {duration/1000:.2f}s, Split size: {split_size}s, Overlap: {overlap_size}s")
    
    # List to store paths of split files
    split_files = []
    chunk_info = []
    validation_details = []
    
    for i in range(total_chunks):
        # Calculate base chunk times
        chunk_start = i * chunk_length
        chunk_end = min(chunk_start + chunk_length, duration)
        
        # Calculate overlap times
        overlap_start = max(0, chunk_start - overlap_ms) if i > 0 else chunk_start
        overlap_end = min(chunk_end + overlap_ms, duration) if i < total_chunks - 1 else chunk_end
        
        # Store chunk information
        chunk_info.append({
            'chunk': i + 1,
            'main_start': chunk_start,
            'main_end': chunk_end,
            'overlap_start': overlap_start,
            'overlap_end': overlap_end
        })
        
        logger.debug(f"Processing chunk {i+1:02d}: {overlap_start/1000:.2f}s -> {overlap_end/1000:.2f}s")
        logger.debug(f"Main segment: {chunk_start/1000:.2f}s -> {chunk_end/1000:.2f}s")
        if i > 0:
            logger.debug(f"Left overlap: {(chunk_start - overlap_start)/1000:.2f}s")
        if i < total_chunks - 1:
            logger.debug(f"Right overlap: {(overlap_end - chunk_end)/1000:.2f}s")
        
        try:
            # Extract and export chunk
            chunk = audio[overlap_start:overlap_end]
            output_path = os.path.join(AUDIO_SPLIT_DIR, f"chunk_{i+1:02d}.wav")
            chunk.export(output_path, format="wav")
            split_files.append(output_path)
            logger.debug(f"Successfully exported chunk {i+1:02d}")
        except Exception as e:
            logger.error(f"Failed to process chunk {i+1:02d}: {str(e)}")
    
    # Enhanced overlap verification
    logger.info("Performing enhanced overlap verification")
    consistent_count = 0
    inconsistent_count = 0
    
    for i in range(len(chunk_info) - 1):
        current = chunk_info[i]
        next_chunk = chunk_info[i + 1]
        
        # Calculate actual overlap duration
        overlap_duration = current['overlap_end'] - next_chunk['overlap_start']
        
        # Determine expected overlap based on chunk position
        is_final_overlap = i == len(chunk_info) - 2
        expected_overlap = overlap_ms
        
        if not is_final_overlap:
            expected_overlap = overlap_ms * 2
        
        # Check if this is a partial overlap due to file end
        is_partial = current['overlap_end'] == duration or next_chunk['overlap_start'] == 0
        
        # Validate overlap with appropriate tolerance
        tolerance = 1  # 1ms tolerance for normal overlaps
        if is_partial:
            tolerance = overlap_ms * 0.1  # 10% tolerance for partial overlaps
        
        is_consistent = abs(overlap_duration - expected_overlap) < tolerance
        
        # Store validation result
        validation_details.append({
            'chunk_pair': (i+1, i+2),
            'is_consistent': is_consistent,
            'overlap_duration': overlap_duration,
            'expected_overlap': expected_overlap,
            'is_partial': is_partial,
            'is_final': is_final_overlap
        })
        
        if is_consistent:
            consistent_count += 1
            logger.debug(f"Overlap between chunks {i+1} and {i+2} is consistent")
        else:
            inconsistent_count += 1
            logger.warning(f"Overlap between chunks {i+1} and {i+2} is inconsistent")
            logger.debug(f"Expected: {expected_overlap/1000:.2f}s, Actual: {overlap_duration/1000:.2f}s")
            logger.debug(f"Type: {'Final Overlap' if is_final_overlap else 'Standard Overlap'}")
            if is_partial:
                logger.debug("(Partial overlap at file boundary)")
            logger.debug(f"Deviation: {abs(overlap_duration - expected_overlap)/1000:.3f}s")
    
    # Generate summary
    summary = (
        f"Splitting Summary:\n"
        f"Total chunks created: {total_chunks}\n"
        f"Overlaps validated: {len(chunk_info) - 1}\n"
        f"Consistent overlaps: {consistent_count}\n"
        f"Inconsistent overlaps: {inconsistent_count} << Expected\n"
    )
    
    logger.info(summary)
    
    # Save the split state
    save_split_state(
        original_file=os.path.basename(file_path),
        split_size=split_size,
        overlap_size=overlap_size,
        split_files=split_files,
        total_duration=duration/1000  # Convert from milliseconds to seconds
    )
    
    return split_files, summary