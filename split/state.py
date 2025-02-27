import os
import json
import shutil
from datetime import datetime
from utils.logger import logger

CHUNKS_DIR = "audio_split"
STATE_FILE = os.path.join(CHUNKS_DIR, "split_state.json")
BACKUP_DIR = os.path.join(CHUNKS_DIR, "backup")

def save_split_state(original_file, split_size, overlap_size, split_files, total_duration=None, chunk_info=None, validation_details=None):
    """Save the state of an audio splitting operation to a JSON file.
    
    Args:
        original_file (str): Name of the original audio file
        split_size (int): Size of each chunk in seconds
        overlap_size (int): Size of overlap between chunks in seconds
        split_files (list): List of paths to the split audio files
        total_duration (float, optional): Total duration of the audio in seconds
        chunk_info (list, optional): List of dictionaries containing timing information for each chunk
        validation_details (list, optional): List of dictionaries containing validation results
    """
    logger.info(f"Saving split state for {len(split_files)} chunks")
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    
    # Backup existing state file if it exists
    if os.path.exists(STATE_FILE):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup_path = os.path.join(BACKUP_DIR, f"split_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            shutil.copy2(STATE_FILE, backup_path)
            logger.debug(f"Previous state file backed up to {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to backup previous state file: {str(e)}")
    
    # Convert milliseconds to seconds for all timecodes in chunk_info
    processed_chunk_info = []
    if chunk_info:
        for chunk in chunk_info:
            processed_chunk_info.append({
                'chunk_number': chunk['chunk'],
                'main_segment': {
                    'start': chunk['main_start'] / 1000,  # Convert to seconds
                    'end': chunk['main_end'] / 1000
                },
                'with_overlap': {
                    'start': chunk['overlap_start'] / 1000,
                    'end': chunk['overlap_end'] / 1000
                },
                'duration': {
                    'main': (chunk['main_end'] - chunk['main_start']) / 1000,
                    'with_overlap': (chunk['overlap_end'] - chunk['overlap_start']) / 1000
                },
                'overlap_before': (chunk['main_start'] - chunk['overlap_start']) / 1000 if chunk['overlap_start'] < chunk['main_start'] else 0,
                'overlap_after': (chunk['overlap_end'] - chunk['main_end']) / 1000 if chunk['overlap_end'] > chunk['main_end'] else 0
            })
    
    # Process validation details
    processed_validation = []
    if validation_details:
        for detail in validation_details:
            processed_detail = {
                'chunk_pair': detail['chunk_pair'],
                'is_consistent': detail['is_consistent']
            }
            
            # Add appropriate fields based on whether there's a gap or overlap
            if detail.get('has_gap', False):
                processed_detail['has_gap'] = True
                processed_detail['gap_duration'] = detail['gap_duration'] / 1000  # Convert to seconds
            else:
                processed_detail['overlap_duration'] = detail.get('overlap_duration', 0) / 1000
                processed_detail['expected_overlap'] = detail.get('expected_overlap', 0) / 1000
                processed_detail['is_partial'] = detail.get('is_partial', False)
                
                # Calculate deviation if applicable
                if 'overlap_duration' in detail and 'expected_overlap' in detail:
                    processed_detail['deviation'] = abs(detail['overlap_duration'] - detail['expected_overlap']) / 1000
            
            processed_validation.append(processed_detail)
    
    # Create the state object
    state = {
        "original_file": original_file,
        "split_timestamp": datetime.now().isoformat(),
        "settings": {
            "chunk_length": split_size,
            "overlap_size": overlap_size
        },
        "chunks": {
            "total_count": len(split_files),
            "files": [{
                "filename": os.path.basename(f),
                "chunk_number": i + 1,
                "timecodes": processed_chunk_info[i] if i < len(processed_chunk_info) else None
            } for i, f in enumerate(split_files)]
        },
        "validation": {
            "performed": validation_details is not None,
            "details": processed_validation if validation_details else None,
            "summary": {
                "total_validations": len(processed_validation) if processed_validation else 0,
                "consistent_count": sum(1 for d in processed_validation if d.get('is_consistent', False)) if processed_validation else 0,
                "gap_count": sum(1 for d in processed_validation if d.get('has_gap', False)) if processed_validation else 0
            } if validation_details else None
        },
        "total_duration": total_duration,
        "version": "2.0"  # Add version to track state format changes
    }
    
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Split state saved to {STATE_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save split state: {str(e)}")
        return False

def get_split_state():
    """Get the current state of audio splits.
    
    Returns:
        dict: The current split state or None if no state file exists
    """
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
                logger.debug(f"Split state loaded from {STATE_FILE}")
                
                # Check if we need to upgrade the state format
                if 'version' not in state:
                    logger.info("Upgrading split state format to latest version")
                    state = _upgrade_state_format(state)
                
                return state
        else:
            logger.debug("No split state file found")
    except Exception as e:
        logger.error(f"Failed to read split state: {str(e)}")
    return None

def _upgrade_state_format(old_state):
    """Upgrade an old state format to the current version.
    
    Args:
        old_state (dict): The state in the old format
        
    Returns:
        dict: The state in the current format
    """
    # Create a copy to avoid modifying the original
    state = old_state.copy()
    
    # Add version
    state['version'] = "2.0"
    
    # Add validation section if missing
    if 'validation' not in state:
        state['validation'] = {
            "performed": False,
            "details": None,
            "summary": None
        }
    
    # Update chunk timecodes if needed
    if 'chunks' in state and 'files' in state['chunks']:
        for chunk in state['chunks']['files']:
            if 'timecodes' in chunk and chunk['timecodes']:
                # Add duration if missing
                if 'duration' not in chunk['timecodes']:
                    main_start = chunk['timecodes']['main_segment']['start']
                    main_end = chunk['timecodes']['main_segment']['end']
                    overlap_start = chunk['timecodes']['with_overlap']['start']
                    overlap_end = chunk['timecodes']['with_overlap']['end']
                    
                    chunk['timecodes']['duration'] = {
                        'main': main_end - main_start,
                        'with_overlap': overlap_end - overlap_start
                    }
    
    logger.debug("State format upgraded successfully")
    return state

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
    
    # Basic summary
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
    
    # Add validation summary if available
    if state.get('validation', {}).get('performed', False):
        validation = state['validation']
        summary += (
            f"\nValidation Results:\n"
            f"- Validations Performed: {validation['summary']['total_validations']}\n"
            f"- Consistent Overlaps: {validation['summary']['consistent_count']}\n"
        )
        
        # Add gap information if available
        if 'gap_count' in validation['summary']:
            summary += f"- Gaps Detected: {validation['summary']['gap_count']}\n"
    
    return summary

def verify_state_integrity():
    """Verify the integrity of the split state file.
    
    Returns:
        tuple: (bool indicating success, verification message)
    """
    logger.info("Verifying split state integrity")
    
    try:
        state = get_split_state()
        if not state:
            return False, "No split state file found"
        
        # Check for required fields
        required_fields = ['original_file', 'split_timestamp', 'settings', 'chunks']
        missing_fields = [field for field in required_fields if field not in state]
        
        if missing_fields:
            logger.warning(f"Split state missing required fields: {', '.join(missing_fields)}")
            return False, f"Split state file is invalid: missing {', '.join(missing_fields)}"
        
        # Check if chunk files exist
        missing_files = []
        for chunk in state['chunks']['files']:
            filename = chunk['filename']
            filepath = os.path.join(CHUNKS_DIR, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        if missing_files:
            logger.warning(f"Split state references {len(missing_files)} missing files")
            return False, f"Split state integrity check failed: {len(missing_files)} referenced files are missing"
        
        logger.info("Split state integrity verified successfully")
        return True, "Split state integrity verified successfully"
        
    except Exception as e:
        logger.error(f"Error verifying split state integrity: {str(e)}")
        return False, f"Error verifying split state integrity: {str(e)}"