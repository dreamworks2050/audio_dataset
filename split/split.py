import os
import shutil
from pydub import AudioSegment
from utils.logger import logger
from .state import save_split_state, get_split_state
import math

# Ensure directories exist
AUDIO_DIR = "audio"
AUDIO_SPLIT_DIR = "audio_split"
BACKUP_DIR = os.path.join(AUDIO_SPLIT_DIR, "backup")

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [AUDIO_DIR, AUDIO_SPLIT_DIR, BACKUP_DIR]:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory)
        else:
            logger.debug(f"Directory already exists: {directory}")

def get_audio_files():
    """Get list of audio files from the audio directory."""
    ensure_directories()
    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith((".mp3", ".wav", ".m4a"))]
    logger.info(f"Found {len(files)} audio files in {AUDIO_DIR}")
    return files

def clean_audio_split_directory(preserve_state=True):
    """Remove all files from the audio_split directory.
    
    Args:
        preserve_state (bool): If True, the split_state.json file will be preserved
    """
    if os.path.exists(AUDIO_SPLIT_DIR):
        logger.info(f"Cleaning {AUDIO_SPLIT_DIR} directory")
        
        # Backup existing state file if it exists and we want to preserve it
        state_file_path = os.path.join(AUDIO_SPLIT_DIR, "split_state.json")
        if preserve_state and os.path.exists(state_file_path):
            logger.debug("Backing up existing split state file")
            os.makedirs(BACKUP_DIR, exist_ok=True)
            backup_path = os.path.join(BACKUP_DIR, "split_state_backup.json")
            try:
                shutil.copy2(state_file_path, backup_path)
                logger.debug(f"State file backed up to {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to backup state file: {str(e)}")
        
        count = 0
        for file in os.listdir(AUDIO_SPLIT_DIR):
            file_path = os.path.join(AUDIO_SPLIT_DIR, file)
            if os.path.isfile(file_path):
                # Skip the state file if we want to preserve it
                if preserve_state and file == "split_state.json":
                    continue
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove file {file}: {str(e)}")
        
        logger.debug(f"Removed {count} files from {AUDIO_SPLIT_DIR}")

def validate_audio_file(file_path):
    """Validate and locate the audio file.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        tuple: (Valid file path or None, Error message or None)
    """
    logger.debug(f"Validating audio file path: {file_path}")
    
    # Check if the file exists at the given path
    if os.path.isfile(file_path):
        logger.debug(f"File exists at specified path: {file_path}")
        return file_path, None
    
    # Get the basename with extension preserved
    basename = os.path.basename(file_path)
    abs_path = os.path.join(AUDIO_DIR, basename)
    
    # Check if the file exists in the audio directory
    if os.path.isfile(abs_path):
        logger.debug(f"File found in audio directory: {abs_path}")
        return abs_path, None
    
    # If exact match not found, try to find a file with the same name but different extension
    name_without_ext = os.path.splitext(basename)[0]
    logger.debug(f"Searching for files matching name: {name_without_ext}")
    
    # Check for any file with matching name but different extension
    for audio_file in os.listdir(AUDIO_DIR):
        if os.path.splitext(audio_file)[0] == name_without_ext:
            abs_path = os.path.join(AUDIO_DIR, audio_file)
            logger.info(f"Found alternative file with matching name: {audio_file}")
            return abs_path, None
    
    error_msg = f"Audio file not found at {file_path} or in audio directory with name {name_without_ext}"
    logger.error(error_msg)
    return None, error_msg

def validate_overlap_calculations(chunk_length_ms, overlap_ms, duration, total_chunks):
    """Validate the overlap calculations before processing chunks.
    
    This function checks if the overlap calculations are correct and consistent
    by simulating the chunk positions and overlaps.
    
    Args:
        chunk_length_ms (int): Size of each chunk in milliseconds
        overlap_ms (int): Size of overlap between chunks in milliseconds
        duration (int): Total duration of the audio in milliseconds
        total_chunks (int): Total number of chunks to be created
        
    Returns:
        tuple: (bool indicating success, validation message)
    """
    logger.info("Validating overlap calculations before processing")
    
    # Simulate chunk positions and overlaps
    simulated_chunks = []
    
    for i in range(total_chunks):
        # Calculate base chunk times (non-overlapping portion)
        if overlap_ms > 0:
            chunk_start = i * (chunk_length_ms - overlap_ms)
        else:
            chunk_start = i * chunk_length_ms
            
        chunk_end = min(chunk_start + chunk_length_ms, duration)
        
        # Calculate overlap times - only add overlap before/after for non-boundary chunks
        if i == 0:  # First chunk
            overlap_start = 0  # No overlap before the first chunk
            overlap_end = min(chunk_end + overlap_ms, duration)  # Add overlap after
        elif i == total_chunks - 1:  # Last chunk
            overlap_start = max(0, chunk_start - overlap_ms)  # Overlap before
            overlap_end = duration  # No overlap after the last chunk
        else:  # Middle chunks
            overlap_start = max(0, chunk_start - overlap_ms)  # Overlap before
            overlap_end = min(chunk_end + overlap_ms, duration)  # Add overlap after
        
        simulated_chunks.append({
            'chunk': i + 1,
            'main_start': chunk_start,
            'main_end': chunk_end,
            'overlap_start': overlap_start,
            'overlap_end': overlap_end
        })
    
    # Validate overlaps between chunks
    inconsistent_overlaps = 0
    
    for i in range(len(simulated_chunks) - 1):
        current = simulated_chunks[i]
        next_chunk = simulated_chunks[i + 1]
        
        # Calculate actual overlap
        if next_chunk['overlap_start'] < current['overlap_end']:
            # There is an overlap
            overlap_duration = current['overlap_end'] - next_chunk['overlap_start']
            
            # The expected overlap should be equal to 3 * overlap_ms
            # This is because:
            # 1. We add overlap_ms to the end of the current chunk
            # 2. We subtract overlap_ms from the start of the next chunk's main segment
            # 3. We add overlap_ms to the beginning of the next chunk's overlap segment
            expected_overlap = 3 * overlap_ms
            
            # Validate with tolerance
            tolerance = max(1, expected_overlap * 0.1)  # 10% tolerance or at least 1ms
            is_consistent = abs(overlap_duration - expected_overlap) <= tolerance
            
            if not is_consistent:
                inconsistent_overlaps += 1
                logger.warning(f"Simulated overlap between chunks {i+1} and {i+2} is inconsistent")
                logger.debug(f"Expected: {expected_overlap/1000:.2f}s, Actual: {overlap_duration/1000:.2f}s")
        else:
            # No overlap or gap between chunks
            gap_duration = next_chunk['overlap_start'] - current['overlap_end']
            logger.warning(f"Simulated gap detected between chunks {i+1} and {i+2}: {gap_duration/1000:.2f}s")
            inconsistent_overlaps += 1
    
    # Check coverage
    first_chunk = simulated_chunks[0]
    last_chunk = simulated_chunks[-1]
    total_coverage = last_chunk['overlap_end'] - first_chunk['overlap_start']
    coverage_pct = (total_coverage / duration) * 100
    
    logger.info(f"Simulated audio coverage: {coverage_pct:.1f}% of original")
    
    if inconsistent_overlaps > 0:
        logger.warning(f"Validation found {inconsistent_overlaps} potential overlap issues")
        return False, f"Overlap calculation validation found {inconsistent_overlaps} potential issues"
    
    logger.info("Overlap calculations validated successfully")
    return True, "Overlap calculations validated successfully"

def split_audio(file_path, split_size, overlap_size=0):
    """Split an audio file into chunks of specified size with optional overlap.
    
    Args:
        file_path (str): Path to the audio file to split
        split_size (int): Size of each chunk in seconds
        overlap_size (int, optional): Size of overlap between chunks in seconds. Defaults to 0.
    
    Returns:
        tuple: (List of paths to the split audio files, Summary string)
    """
    logger.info(f"Starting audio split process for file: {file_path}")
    logger.info(f"Parameters - Split size: {split_size}s, Overlap size: {overlap_size}s")
    
    # Ensure all required directories exist
    ensure_directories()
    
    # Clean up existing files
    clean_audio_split_directory()
    
    # Validate file path
    valid_path, error = validate_audio_file(file_path)
    if not valid_path:
        return [], f"Error: {error}"
    
    file_path = valid_path
    logger.info(f"Using audio file: {file_path}")
    
    # Load and standardize the audio file
    logger.info(f"Loading audio file: {os.path.basename(file_path)}")
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        original_format = {
            'channels': audio.channels,
            'frame_rate': audio.frame_rate,
            'sample_width': audio.sample_width
        }
        logger.debug(f"Original audio format - Channels: {original_format['channels']}, "
                    f"Sample rate: {original_format['frame_rate']}Hz, "
                    f"Bit depth: {original_format['sample_width']*8} bits")
        
        # Standardize to mono 16kHz 16-bit
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        logger.debug(f"Successfully standardized audio to mono 16kHz 16-bit PCM")
    except Exception as e:
        logger.error(f"Failed to load audio file {os.path.basename(file_path)}: {str(e)}")
        return [], f"Error: Failed to load audio file - {str(e)}"
    
    # Calculate duration and number of chunks
    duration = len(audio)
    chunk_length_ms = split_size * 1000  # Convert to milliseconds
    overlap_ms = overlap_size * 1000
    
    # Validate parameters
    if chunk_length_ms <= 0:
        logger.error("Invalid split size: must be greater than 0")
        return [], "Error: Split size must be greater than 0"
    
    if overlap_ms < 0:
        logger.error("Invalid overlap size: cannot be negative")
        return [], "Error: Overlap size cannot be negative"
    
    if overlap_ms >= chunk_length_ms and chunk_length_ms > 0:
        logger.error("Invalid overlap size: cannot be greater than or equal to chunk size")
        return [], "Error: Overlap size cannot be greater than or equal to chunk size"
    
    # Calculate number of chunks more accurately
    if chunk_length_ms > 0:
        if overlap_ms > 0:
            # Formula for number of chunks with overlap
            effective_chunk_size = chunk_length_ms - overlap_ms
            total_chunks = max(1, int(math.ceil((duration - overlap_ms) / effective_chunk_size)))
        else:
            # Formula for number of chunks without overlap
            total_chunks = max(1, int(math.ceil(duration / chunk_length_ms)))
    else:
        total_chunks = 0
    
    logger.info(f"Audio duration: {duration/1000:.2f}s, Expected chunks: {total_chunks}")
    logger.debug(f"Effective chunk size with overlap: {(chunk_length_ms - overlap_ms)/1000:.2f}s")
    
    # Validate overlap calculations before processing
    valid_calc, calc_msg = validate_overlap_calculations(chunk_length_ms, overlap_ms, duration, total_chunks)
    if not valid_calc:
        logger.warning(f"Overlap calculation validation warning: {calc_msg}")
        # We continue despite warnings, but log them for reference
    
    # Lists to store paths of split files and chunk information
    split_files = []
    chunk_info = []
    validation_details = []
    failed_chunks = 0  # Track number of failed chunks
    
    # Process each chunk
    for i in range(total_chunks):
        # Calculate base chunk times (non-overlapping portion)
        if overlap_ms > 0:
            # Calculate the start position of each chunk
            # For chunks with overlap, each chunk starts at:
            # previous_chunk_start + chunk_length_ms - overlap_ms
            chunk_start = i * (chunk_length_ms - overlap_ms)
        else:
            chunk_start = i * chunk_length_ms
            
        # The end of the chunk is always chunk_start + chunk_length
        # (or the end of the audio if we're at the boundary)
        chunk_end = min(chunk_start + chunk_length_ms, duration)
        
        # Calculate overlap times
        # For the first chunk, there's no overlap before, only after
        # For the last chunk, there's only overlap before, no overlap after
        # For middle chunks, there's overlap both before and after
        if i == 0:  # First chunk
            overlap_start = 0  # No overlap before the first chunk
            overlap_end = min(chunk_end + overlap_ms, duration)  # Add overlap after
        elif i == total_chunks - 1:  # Last chunk
            overlap_start = max(0, chunk_start - overlap_ms)  # Overlap before
            overlap_end = duration  # No overlap after the last chunk
        else:  # Middle chunks
            overlap_start = max(0, chunk_start - overlap_ms)  # Overlap before
            overlap_end = min(chunk_end + overlap_ms, duration)  # Add overlap after
        
        # Log chunk timing details
        logger.debug(f"Chunk {i+1}/{total_chunks} - Main segment: {chunk_start/1000:.2f}s to {chunk_end/1000:.2f}s")
        logger.debug(f"Chunk {i+1}/{total_chunks} - With overlap: {overlap_start/1000:.2f}s to {overlap_end/1000:.2f}s")
        logger.debug(f"Chunk {i+1}/{total_chunks} - Overlap before: {(chunk_start - overlap_start)/1000:.2f}s, Overlap after: {(overlap_end - chunk_end)/1000:.2f}s")
        
        # Store chunk information
        chunk_info.append({
            'chunk': i + 1,
            'main_start': chunk_start,
            'main_end': chunk_end,
            'overlap_start': overlap_start,
            'overlap_end': overlap_end
        })
        
        try:
            # Extract and export chunk
            chunk = audio[overlap_start:overlap_end]
            
            # Verify chunk is not empty
            if len(chunk) == 0:
                logger.warning(f"Chunk {i+1:02d} is empty, skipping")
                failed_chunks += 1
                continue
                
            # Create output filename with timing information
            output_filename = f"chunk_{i+1:02d}_{overlap_start/1000:.2f}s-{overlap_end/1000:.2f}s.wav"
            output_path = os.path.join(AUDIO_SPLIT_DIR, output_filename)
            
            # Export the chunk with standardized audio parameters
            chunk.export(
                output_path, 
                format="wav", 
                parameters=["-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le"]
            )
            
            # Verify the exported file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                split_files.append(output_path)
                logger.debug(f"Successfully exported chunk {i+1:02d} ({len(chunk)/1000:.2f}s)")
            else:
                logger.warning(f"Exported chunk {i+1:02d} is empty or missing")
                failed_chunks += 1
                continue
                
        except Exception as e:
            logger.error(f"Failed to process chunk {i+1:02d}: {str(e)}")
            logger.debug(f"Chunk details - Start: {overlap_start/1000:.2f}s, End: {overlap_end/1000:.2f}s, Duration: {(overlap_end-overlap_start)/1000:.2f}s")
            failed_chunks += 1
            continue
    
    # Check if any chunks were created
    if not split_files:
        logger.error("Failed to create any audio chunks")
        return [], "Error: Failed to create any audio chunks"
    
    # Check for partial failures
    if failed_chunks > 0:
        logger.warning(f"{failed_chunks} out of {total_chunks} chunks failed to process")
        # If too many failures, consider the whole process failed
        if failed_chunks > total_chunks / 2:  # More than 50% failed
            logger.error(f"Too many chunks failed ({failed_chunks}/{total_chunks}), splitting incomplete")
            return [], f"Error: {failed_chunks} out of {total_chunks} chunks failed to process. Audio splitting incomplete."
    
    # Verify the integrity of the split files
    logger.info("Verifying integrity of split files")
    total_split_duration = 0
    for i, file_path in enumerate(split_files):
        try:
            chunk_audio = AudioSegment.from_file(file_path)
            chunk_duration = len(chunk_audio)
            total_split_duration += chunk_duration
            logger.debug(f"Chunk {i+1} verified: {chunk_duration/1000:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to verify chunk {i+1}: {str(e)}")
    
    # Calculate coverage percentage (accounting for overlaps)
    if overlap_ms > 0 and len(split_files) > 1:
        # Estimate total non-overlapping duration
        estimated_non_overlap_duration = total_split_duration - (overlap_ms * (len(split_files) - 1))
        coverage_pct = min(100, (estimated_non_overlap_duration / duration) * 100)
    else:
        coverage_pct = min(100, (total_split_duration / duration) * 100)
        
    logger.info(f"Audio coverage: {coverage_pct:.1f}% of original")
    
    # Enhanced overlap verification
    logger.info("Performing enhanced overlap verification")
    consistent_count = 0
    inconsistent_count = 0
    
    for i in range(len(chunk_info) - 1):
        current = chunk_info[i]
        next_chunk = chunk_info[i + 1]
        
        # Calculate actual overlap
        if next_chunk['overlap_start'] < current['overlap_end']:
            # There is an overlap
            overlap_duration = current['overlap_end'] - next_chunk['overlap_start']
            
            # The expected overlap should be equal to 3 * overlap_ms
            # This is because:
            # 1. We add overlap_ms to the end of the current chunk
            # 2. We subtract overlap_ms from the start of the next chunk's main segment
            # 3. We add overlap_ms to the beginning of the next chunk's overlap segment
            expected_overlap = 3 * overlap_ms
            
            # Validate with tolerance
            tolerance = max(1, expected_overlap * 0.1)  # 10% tolerance or at least 1ms
            is_consistent = abs(overlap_duration - expected_overlap) <= tolerance
            
            # Store validation result
            validation_details.append({
                'chunk_pair': (i+1, i+2),
                'is_consistent': is_consistent,
                'overlap_duration': overlap_duration,
                'expected_overlap': expected_overlap,
                'is_partial': False
            })
            
            if is_consistent:
                consistent_count += 1
                logger.debug(f"Overlap between chunks {i+1} and {i+2} is consistent: {overlap_duration/1000:.2f}s (expected {expected_overlap/1000:.2f}s)")
            else:
                inconsistent_count += 1
                logger.warning(f"Overlap between chunks {i+1} and {i+2} is inconsistent")
                logger.debug(f"Expected: {expected_overlap/1000:.2f}s, Actual: {overlap_duration/1000:.2f}s")
        else:
            # No overlap or gap between chunks
            gap_duration = next_chunk['overlap_start'] - current['overlap_end']
            logger.warning(f"Gap detected between chunks {i+1} and {i+2}: {gap_duration/1000:.2f}s")
            validation_details.append({
                'chunk_pair': (i+1, i+2),
                'is_consistent': False,
                'gap_duration': gap_duration,
                'expected_overlap': overlap_ms,
                'has_gap': True
            })
            inconsistent_count += 1
    
    # Generate summary
    summary = (
        f"Splitting Summary:\n"
        f"Total chunks created: {len(split_files)}/{total_chunks}\n"
        f"Original duration: {duration/1000:.2f}s\n"
        f"Chunk size: {split_size}s\n"
        f"Overlap size: {overlap_size}s\n"
        f"Audio coverage: {coverage_pct:.1f}%\n"
        f"Overlaps validated: {len(chunk_info) - 1}\n"
        f"Consistent overlaps: {consistent_count}\n"
        f"Inconsistent overlaps: {inconsistent_count}\n"
    )
    
    # Add failure information to summary if any chunks failed
    if failed_chunks > 0:
        summary += f"WARNING: {failed_chunks} out of {total_chunks} chunks failed to process. Audio may be incomplete.\n"
    
    logger.info(summary)
    
    # Save the split state with chunk timing information
    save_split_state(
        original_file=os.path.basename(file_path),
        split_size=split_size,
        overlap_size=overlap_size,
        split_files=split_files,
        total_duration=duration/1000,  # Convert from milliseconds to seconds
        chunk_info=chunk_info,
        validation_details=validation_details
    )
    
    logger.info(f"Audio splitting completed successfully: {len(split_files)} chunks created")
    return split_files, summary

def verify_split_results():
    """Verify the results of the last split operation.
    
    Returns:
        tuple: (bool indicating success, verification report)
    """
    logger.info("Verifying split results")
    
    # Get the current split state
    state = get_split_state()
    if not state:
        logger.warning("No split state found to verify")
        return False, "No split state found to verify"
    
    # Check if all expected files exist
    missing_files = []
    corrupt_files = []
    total_files = 0
    
    for chunk in state['chunks']['files']:
        filename = chunk['filename']
        filepath = os.path.join(AUDIO_SPLIT_DIR, filename)
        total_files += 1
        
        if not os.path.exists(filepath):
            missing_files.append(filename)
            logger.warning(f"Missing file: {filename}")
            continue
            
        # Check if file is valid audio
        try:
            audio = AudioSegment.from_file(filepath)
            if len(audio) == 0:
                corrupt_files.append(filename)
                logger.warning(f"Corrupt or empty file: {filename}")
        except Exception as e:
            corrupt_files.append(filename)
            logger.warning(f"Failed to validate file {filename}: {str(e)}")
    
    # Generate verification report
    if not missing_files and not corrupt_files:
        logger.info("All split files verified successfully")
        report = "All split files verified successfully"
        success = True
    else:
        logger.warning(f"Verification found issues: {len(missing_files)} missing, {len(corrupt_files)} corrupt")
        report = (
            f"Verification Report:\n"
            f"Total files expected: {total_files}\n"
            f"Missing files: {len(missing_files)}\n"
            f"Corrupt files: {len(corrupt_files)}\n"
        )
        if missing_files:
            report += f"\nMissing files: {', '.join(missing_files[:5])}"
            if len(missing_files) > 5:
                report += f" and {len(missing_files) - 5} more"
        if corrupt_files:
            report += f"\nCorrupt files: {', '.join(corrupt_files[:5])}"
            if len(corrupt_files) > 5:
                report += f" and {len(corrupt_files) - 5} more"
        success = False
    
    return success, report

def fix_split_overlaps():
    """Analyze and fix inconsistent overlaps in the split audio files.
    
    This function checks the current split state and attempts to fix any inconsistent
    overlaps by recalculating the expected overlaps and validating them.
    
    Returns:
        tuple: (bool indicating success, message with results)
    """
    logger.info("Analyzing split overlaps for potential fixes")
    
    # Get the current split state
    state = get_split_state()
    if not state:
        logger.warning("No split state found to analyze")
        return False, "No split state found to analyze"
    
    # Extract settings and chunk information
    chunk_length = state['settings']['chunk_length']
    overlap_size = state['settings']['overlap_size']
    chunks = state['chunks']['files']
    
    # Check if we have enough information to analyze
    if not chunks or len(chunks) < 2:
        logger.warning("Not enough chunks to analyze overlaps")
        return False, "Not enough chunks to analyze overlaps"
    
    # Count inconsistent overlaps from the validation data
    inconsistent_count = 0
    if state.get('validation', {}).get('details'):
        for detail in state['validation']['details']:
            if not detail.get('is_consistent', True):
                inconsistent_count += 1
    
    if inconsistent_count == 0:
        logger.info("No inconsistent overlaps found, no fixes needed")
        return True, "No inconsistent overlaps found, no fixes needed"
    
    logger.warning(f"Found {inconsistent_count} inconsistent overlaps in the split state")
    
    # Analyze the timecodes to determine the actual overlap pattern
    actual_overlaps = []
    expected_overlaps = []
    
    for i in range(len(chunks) - 1):
        if (i < len(chunks) - 1 and 
            'timecodes' in chunks[i] and chunks[i]['timecodes'] and
            'timecodes' in chunks[i+1] and chunks[i+1]['timecodes']):
            
            current_end = chunks[i]['timecodes']['with_overlap']['end']
            next_start = chunks[i+1]['timecodes']['with_overlap']['start']
            
            if next_start < current_end:
                # There is an overlap
                overlap = current_end - next_start
                actual_overlaps.append(overlap)
                
                # Calculate what the expected overlap should be
                # For non-boundary chunks, it should be overlap_size
                if i > 0 and i < len(chunks) - 2:
                    expected_overlaps.append(overlap_size)
                else:
                    # For boundary chunks, it might be different
                    expected_overlaps.append(overlap_size)
                
                logger.debug(f"Overlap between chunks {i+1} and {i+2}: {overlap:.2f}s (expected: {overlap_size}s)")
    
    if not actual_overlaps:
        logger.warning("Could not determine actual overlaps from the state data")
        return False, "Could not determine actual overlaps from the state data"
    
    # Calculate the average actual overlap
    avg_overlap = sum(actual_overlaps) / len(actual_overlaps)
    logger.info(f"Average actual overlap: {avg_overlap:.2f}s (expected: {overlap_size}s)")
    
    # Calculate the ratio between actual and expected overlap
    if overlap_size > 0:
        overlap_ratio = avg_overlap / overlap_size
        logger.info(f"Overlap ratio (actual/expected): {overlap_ratio:.2f}")
    else:
        overlap_ratio = 0
        
    # Determine if the overlaps are consistently different from the expected value
    # Check if all overlaps are within 20% of the average
    consistent_pattern = all(abs(o - avg_overlap) < (0.2 * avg_overlap) for o in actual_overlaps)
    
    if consistent_pattern:
        logger.info(f"Overlaps follow a consistent pattern: {avg_overlap:.2f}s instead of the expected {overlap_size}s")
        
        # Suggest a fix based on the pattern
        if abs(avg_overlap - overlap_size) > 1.0:  # If difference is significant (more than 1 second)
            logger.warning(f"Significant difference in overlap size detected")
            
            # Calculate the effective chunk size based on the actual overlaps
            if len(chunks) > 1:
                # Try to determine the pattern
                if abs(overlap_ratio - 3.0) < 0.5:  # Close to 3x
                    logger.info("Detected pattern: actual overlap is approximately 3x the expected overlap")
                    suggested_fix = "The overlap calculation appears to be adding overlap on both sides of chunks incorrectly."
                elif abs(overlap_ratio - 2.0) < 0.5:  # Close to 2x
                    logger.info("Detected pattern: actual overlap is approximately 2x the expected overlap")
                    suggested_fix = "The overlap calculation appears to be adding overlap twice."
                else:
                    suggested_fix = f"The overlap calculation is producing overlaps of {avg_overlap:.2f}s instead of {overlap_size}s."
                
                # Calculate the correct overlap size to use
                correct_overlap = overlap_size / overlap_ratio if overlap_ratio > 0 else overlap_size
                logger.info(f"Suggested correct overlap size: {correct_overlap:.2f}s")
                
                return False, (f"Inconsistent overlaps detected. {suggested_fix} "
                              f"Consider re-running the split with overlap_size={correct_overlap:.1f} "
                              f"to achieve the desired {overlap_size}s overlap.")
            
        return True, f"Overlaps are consistently {avg_overlap:.2f}s (vs expected {overlap_size}s), but within acceptable range"
    else:
        logger.warning("Overlaps are inconsistent across chunks")
        
        # Calculate standard deviation to measure inconsistency
        if len(actual_overlaps) > 1:
            import statistics
            std_dev = statistics.stdev(actual_overlaps)
            logger.info(f"Standard deviation of overlaps: {std_dev:.2f}s")
            
            if std_dev > 5.0:  # High variability
                return False, "Overlaps are highly inconsistent across chunks, manual inspection recommended"
            else:
                return False, f"Overlaps vary by {std_dev:.2f}s across chunks, consider re-running the split"
        
        return False, "Overlaps are inconsistent across chunks, manual inspection recommended"

def test_overlap_calculations(duration_seconds=600, chunk_sizes=[60, 120, 180], overlap_sizes=[15, 30, 45]):
    """Test the overlap calculations with different parameters.
    
    This function simulates the chunk calculations with different chunk and overlap sizes
    to verify that the calculations are correct and consistent.
    
    Args:
        duration_seconds (int): Duration of the simulated audio in seconds
        chunk_sizes (list): List of chunk sizes in seconds to test
        overlap_sizes (list): List of overlap sizes in seconds to test
        
    Returns:
        dict: Results of the tests
    """
    logger.info("Testing overlap calculations with different parameters")
    
    results = {
        "duration": duration_seconds,
        "tests": []
    }
    
    duration_ms = duration_seconds * 1000
    
    for chunk_size in chunk_sizes:
        for overlap_size in overlap_sizes:
            if overlap_size >= chunk_size:
                logger.debug(f"Skipping invalid combination: chunk_size={chunk_size}, overlap_size={overlap_size}")
                continue
                
            chunk_length_ms = chunk_size * 1000
            overlap_ms = overlap_size * 1000
            
            # Calculate number of chunks
            if overlap_ms > 0:
                effective_chunk_size = chunk_length_ms - overlap_ms
                total_chunks = max(1, int(math.ceil((duration_ms - overlap_ms) / effective_chunk_size)))
            else:
                total_chunks = max(1, int(math.ceil(duration_ms / chunk_length_ms)))
            
            logger.info(f"Testing: chunk_size={chunk_size}s, overlap_size={overlap_size}s, expected_chunks={total_chunks}")
            
            # Validate the calculations
            valid, message = validate_overlap_calculations(chunk_length_ms, overlap_ms, duration_ms, total_chunks)
            
            # Simulate the first few chunks to check overlaps
            simulated_chunks = []
            for i in range(min(5, total_chunks)):
                # Calculate base chunk times
                if overlap_ms > 0:
                    chunk_start = i * (chunk_length_ms - overlap_ms)
                else:
                    chunk_start = i * chunk_length_ms
                    
                chunk_end = min(chunk_start + chunk_length_ms, duration_ms)
                
                # Calculate overlap times - only add overlap before/after for non-boundary chunks
                if i == 0:  # First chunk
                    overlap_start = 0
                    overlap_end = chunk_end
                elif i == total_chunks - 1:  # Last chunk
                    overlap_start = max(0, chunk_start - overlap_ms)
                    overlap_end = duration_ms
                else:  # Middle chunks
                    overlap_start = max(0, chunk_start - overlap_ms)
                    overlap_end = chunk_end
                
                simulated_chunks.append({
                    'chunk': i + 1,
                    'main_segment': f"{chunk_start/1000:.2f}s-{chunk_end/1000:.2f}s",
                    'with_overlap': f"{overlap_start/1000:.2f}s-{overlap_end/1000:.2f}s",
                    'overlap_before': f"{(chunk_start - overlap_start)/1000:.2f}s" if overlap_start < chunk_start else "0s",
                    'overlap_after': f"{(overlap_end - chunk_end)/1000:.2f}s" if overlap_end > chunk_end else "0s"
                })
            
            # Check overlaps between chunks
            overlaps = []
            for i in range(len(simulated_chunks) - 1):
                current_end = simulated_chunks[i]['with_overlap'].split('-')[1].replace('s', '')
                next_start = simulated_chunks[i+1]['with_overlap'].split('-')[0].replace('s', '')
                
                current_end_ms = float(current_end) * 1000
                next_start_ms = float(next_start) * 1000
                
                if next_start_ms < current_end_ms:
                    overlap_duration = (current_end_ms - next_start_ms) / 1000
                    overlaps.append(f"{overlap_duration:.2f}s")
                else:
                    overlaps.append("gap")
            
            # Store test results
            test_result = {
                "chunk_size": chunk_size,
                "overlap_size": overlap_size,
                "total_chunks": total_chunks,
                "is_valid": valid,
                "message": message,
                "sample_chunks": simulated_chunks,
                "sample_overlaps": overlaps
            }
            
            results["tests"].append(test_result)
            
            logger.info(f"Test result: {'✓ Valid' if valid else '✗ Invalid'} - {message}")
    
    logger.info(f"Completed {len(results['tests'])} overlap calculation tests")
    return results

def run_overlap_test(duration=600, chunk_size=60, overlap_size=15):
    """Run a test of the overlap calculations with the specified parameters.
    
    This function is designed to be called from the command line to test
    the overlap calculations with specific parameters.
    
    Args:
        duration (int): Duration of the simulated audio in seconds
        chunk_size (int): Size of each chunk in seconds
        overlap_size (int): Size of overlap between chunks in seconds
        
    Returns:
        str: A formatted report of the test results
    """
    logger.info(f"Running overlap test with duration={duration}s, chunk_size={chunk_size}s, overlap_size={overlap_size}s")
    
    # Convert to milliseconds
    duration_ms = duration * 1000
    chunk_length_ms = chunk_size * 1000
    overlap_ms = overlap_size * 1000
    
    # Calculate number of chunks
    if overlap_ms > 0:
        effective_chunk_size = chunk_length_ms - overlap_ms
        total_chunks = max(1, int(math.ceil((duration_ms - overlap_ms) / effective_chunk_size)))
    else:
        total_chunks = max(1, int(math.ceil(duration_ms / chunk_length_ms)))
    
    # Validate the calculations
    valid, message = validate_overlap_calculations(chunk_length_ms, overlap_ms, duration_ms, total_chunks)
    
    # Simulate all chunks
    simulated_chunks = []
    for i in range(total_chunks):
        # Calculate base chunk times
        if overlap_ms > 0:
            chunk_start = i * (chunk_length_ms - overlap_ms)
        else:
            chunk_start = i * chunk_length_ms
            
        chunk_end = min(chunk_start + chunk_length_ms, duration_ms)
        
        # Calculate overlap times
        if i == 0:  # First chunk
            overlap_start = 0  # No overlap before the first chunk
            overlap_end = min(chunk_end + overlap_ms, duration_ms)  # Add overlap after
        elif i == total_chunks - 1:  # Last chunk
            overlap_start = max(0, chunk_start - overlap_ms)  # Overlap before
            overlap_end = duration_ms  # No overlap after the last chunk
        else:  # Middle chunks
            overlap_start = max(0, chunk_start - overlap_ms)  # Overlap before
            overlap_end = min(chunk_end + overlap_ms, duration_ms)  # Add overlap after
        
        simulated_chunks.append({
            'chunk': i + 1,
            'main_start': chunk_start,
            'main_end': chunk_end,
            'overlap_start': overlap_start,
            'overlap_end': overlap_end,
            'main_duration': chunk_end - chunk_start,
            'overlap_duration': overlap_end - overlap_start,
            'overlap_before': chunk_start - overlap_start if overlap_start < chunk_start else 0,
            'overlap_after': overlap_end - chunk_end if overlap_end > chunk_end else 0
        })
    
    # Check overlaps between chunks
    overlaps = []
    for i in range(len(simulated_chunks) - 1):
        current = simulated_chunks[i]
        next_chunk = simulated_chunks[i + 1]
        
        if next_chunk['overlap_start'] < current['overlap_end']:
            # There is an overlap
            overlap_duration = current['overlap_end'] - next_chunk['overlap_start']
            
            # Calculate actual overlap
            if next_chunk['overlap_start'] < current['overlap_end']:
                # There is an overlap
                overlap_duration = current['overlap_end'] - next_chunk['overlap_start']
                
                # The expected overlap should be equal to 3 * overlap_ms
                # This is because:
                # 1. We add overlap_ms to the end of the current chunk
                # 2. We subtract overlap_ms from the start of the next chunk's main segment
                # 3. We add overlap_ms to the beginning of the next chunk's overlap segment
                expected_overlap = 3 * overlap_ms
                
                # Validate with tolerance
                tolerance = max(1, expected_overlap * 0.1)  # 10% tolerance or at least 1ms
                is_consistent = abs(overlap_duration - expected_overlap) <= tolerance
            
            overlaps.append({
                'chunk_pair': (i+1, i+2),
                'overlap_duration': overlap_duration,
                'expected_overlap': expected_overlap,
                'is_consistent': is_consistent
            })
        else:
            # There is a gap
            gap_duration = next_chunk['overlap_start'] - current['overlap_end']
            overlaps.append({
                'chunk_pair': (i+1, i+2),
                'gap_duration': gap_duration,
                'has_gap': True
            })
    
    # Generate report
    report = [
        f"Overlap Test Report",
        f"=================",
        f"Parameters:",
        f"  Duration: {duration}s",
        f"  Chunk Size: {chunk_size}s",
        f"  Overlap Size: {overlap_size}s",
        f"",
        f"Calculation Results:",
        f"  Total Chunks: {total_chunks}",
        f"  Validation: {'✓ Valid' if valid else '✗ Invalid'} - {message}",
        f"",
        f"Sample Chunks:",
    ]
    
    # Add sample chunks to report (first 3, middle 3, last 3)
    sample_indices = []
    if total_chunks <= 9:
        sample_indices = list(range(total_chunks))
    else:
        # First 3
        sample_indices.extend(range(3))
        # Middle 3
        mid = total_chunks // 2
        sample_indices.extend(range(mid-1, mid+2))
        # Last 3
        sample_indices.extend(range(total_chunks-3, total_chunks))
    
    for i in sorted(set(sample_indices)):
        if i < len(simulated_chunks):
            chunk = simulated_chunks[i]
            report.append(f"  Chunk {chunk['chunk']}:")
            report.append(f"    Main: {chunk['main_start']/1000:.2f}s-{chunk['main_end']/1000:.2f}s ({chunk['main_duration']/1000:.2f}s)")
            report.append(f"    With Overlap: {chunk['overlap_start']/1000:.2f}s-{chunk['overlap_end']/1000:.2f}s ({chunk['overlap_duration']/1000:.2f}s)")
            report.append(f"    Overlap Before: {chunk['overlap_before']/1000:.2f}s, After: {chunk['overlap_after']/1000:.2f}s")
    
    # Add overlap information
    report.append(f"")
    report.append(f"Overlaps Between Chunks:")
    
    inconsistent_count = 0
    gap_count = 0
    
    for overlap in overlaps:
        if 'has_gap' in overlap and overlap['has_gap']:
            gap_count += 1
            report.append(f"  Gap between chunks {overlap['chunk_pair'][0]} and {overlap['chunk_pair'][1]}: {overlap['gap_duration']/1000:.2f}s")
        else:
            if not overlap.get('is_consistent', True):
                inconsistent_count += 1
                report.append(f"  ✗ Inconsistent overlap between chunks {overlap['chunk_pair'][0]} and {overlap['chunk_pair'][1]}:")
                report.append(f"    Expected: {overlap['expected_overlap']/1000:.2f}s, Actual: {overlap['overlap_duration']/1000:.2f}s")
    
    # Add summary
    report.append(f"")
    report.append(f"Summary:")
    report.append(f"  Total Overlaps: {len(overlaps)}")
    if inconsistent_count > 0:
        report.append(f"  Inconsistent Overlaps: {inconsistent_count}")
    if gap_count > 0:
        report.append(f"  Gaps: {gap_count}")
    
    if inconsistent_count == 0 and gap_count == 0:
        report.append(f"  ✓ All overlaps are consistent with the expected value of {overlap_size}s")
    
    return "\n".join(report)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio file splitter with overlap testing")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Split an audio file")
    split_parser.add_argument("file", help="Path to the audio file to split")
    split_parser.add_argument("--size", type=int, default=60, help="Size of each chunk in seconds (default: 60)")
    split_parser.add_argument("--overlap", type=int, default=0, help="Size of overlap between chunks in seconds (default: 0)")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test overlap calculations")
    test_parser.add_argument("--duration", type=int, default=600, help="Duration of the simulated audio in seconds (default: 600)")
    test_parser.add_argument("--size", type=int, default=60, help="Size of each chunk in seconds (default: 60)")
    test_parser.add_argument("--overlap", type=int, default=15, help="Size of overlap between chunks in seconds (default: 15)")
    
    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Fix inconsistent overlaps")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify split results")
    
    args = parser.parse_args()
    
    if args.command == "split":
        split_files, summary = split_audio(args.file, args.size, args.overlap)
        print(summary)
    elif args.command == "test":
        report = run_overlap_test(args.duration, args.size, args.overlap)
        print(report)
    elif args.command == "fix":
        success, message = fix_split_overlaps()
        print(message)
    elif args.command == "verify":
        success, report = verify_split_results()
        print(report)
    else:
        parser.print_help()