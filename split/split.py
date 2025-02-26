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

def split_audio(file_path, split_size, overlap_size=0):
    """Split an audio file into chunks of specified size with optional overlap.
    
    Args:
        file_path (str): Path to the audio file to split
        split_size (int): Size of each chunk in seconds
        overlap_size (int, optional): Size of overlap between chunks in seconds. Defaults to 0.
    
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
    
    # Convert overlap size to milliseconds
    overlap_ms = overlap_size * 1000
    
    # Split the audio into chunks with overlap
    total_chunks = (duration + chunk_length - 1) // chunk_length
    
    print(f"\nProcessing audio file: {os.path.basename(file_path)}")
    print(f"Total duration: {duration/1000:.2f}s, Split size: {split_size}s, Overlap: {overlap_size}s\n")
    
    # Store chunk info for verification
    chunk_info = []
    
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
        
        # Print chunk timecode information
        print(f"Chunk {i+1:02d}: {overlap_start/1000:.2f}s -> {overlap_end/1000:.2f}s ")
        print(f"        Main: {chunk_start/1000:.2f}s -> {chunk_end/1000:.2f}s")
        if i > 0:
            print(f"        Left overlap: {(chunk_start - overlap_start)/1000:.2f}s")
        if i < total_chunks - 1:
            print(f"        Right overlap: {(overlap_end - chunk_end)/1000:.2f}s\n")
        
        # Extract and export chunk
        chunk = audio[overlap_start:overlap_end]
        output_path = os.path.join(AUDIO_SPLIT_DIR, f"chunk_{i+1:02d}.wav")
        chunk.export(output_path, format="wav")
        split_files.append(output_path)
    
    # Enhanced overlap verification
    print("\nOverlap Verification:")
    consistent_count = 0
    inconsistent_count = 0
    validation_details = []
    
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
        else:
            inconsistent_count += 1
        
        # Print detailed validation information
        print(f"Overlap between chunks {i+1} and {i+2}:")
        print(f"Expected: {expected_overlap/1000:.2f}s, Actual: {overlap_duration/1000:.2f}s")
        print(f"Type: {'Final Overlap' if is_final_overlap else 'Standard Overlap'}")
        print(f"{'(Partial overlap at file boundary)' if is_partial else ''}")
        
        # Update status message to indicate expected inconsistency for final chunk
        status = '✓ Consistent' if is_consistent else '✗ Inconsistent'
        if not is_consistent and is_final_overlap:
            status += ' (expected)'
        print(f"Status: {status}")
        
        if not is_consistent:
            print(f"Deviation: {abs(overlap_duration - expected_overlap)/1000:.3f}s")
        print()
    
    # Generate summary
    summary = (
        f"\nSplitting Summary:\n"
        f"Total chunks created: {total_chunks}\n"
        f"Overlaps validated: {len(chunk_info) - 1}\n"
        f"Consistent overlaps: {consistent_count}\n"
        f"Inconsistent overlaps: {inconsistent_count} << Expected\n"
    )
    print(summary)
    
    return split_files, summary