import os
import json
import shutil
from pydub import AudioSegment
from utils.logger import logger
import time
from datetime import datetime

# Ensure directories exist
AUDIO_DIR = "audio"
AUDIO_AI_OPTIMIZED_DIR = "audio_ai_optimized"

class AudioOptimizer:
    """Class to handle AI optimization of audio files by creating multiple versions with different chunk and overlap settings."""
    
    def __init__(self):
        """Initialize the AudioOptimizer."""
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [AUDIO_DIR, AUDIO_AI_OPTIMIZED_DIR]:
            if not os.path.exists(directory):
                logger.info(f"Creating directory: {directory}")
                os.makedirs(directory)
            else:
                logger.debug(f"Directory already exists: {directory}")
    
    def get_audio_files(self):
        """Get list of audio files from the audio directory."""
        self.ensure_directories()
        files = [f for f in os.listdir(AUDIO_DIR) if f.endswith((".mp3", ".wav", ".m4a"))]
        logger.info(f"Found {len(files)} audio files in {AUDIO_DIR}")
        return files
    
    def validate_audio_file(self, file_path):
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
    
    def optimize_audio(self, file_path, chunk_lengths, overlap_lengths, max_duration=None):
        """Optimize audio by creating multiple versions with different chunk and overlap settings.
        
        Args:
            file_path (str): Path to the audio file to optimize
            chunk_lengths (list): List of chunk lengths in seconds
            overlap_lengths (list): List of overlap lengths in seconds
            max_duration (float, optional): Maximum duration in seconds to process. If None, process the entire audio.
        
        Returns:
            tuple: (Success status, Summary string)
        """
        logger.info(f"Starting audio optimization process for file: {file_path}")
        logger.info(f"Parameters - Chunk lengths: {chunk_lengths}, Overlap lengths: {overlap_lengths}, Max duration: {max_duration}")
        
        # Ensure all required directories exist
        self.ensure_directories()
        
        # Validate file path
        valid_path, error = self.validate_audio_file(file_path)
        if not valid_path:
            return False, f"Error: {error}"
        
        file_path = valid_path
        logger.info(f"Using audio file: {file_path}")
        
        # Create a timestamp for this optimization run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a subdirectory for this optimization run
        run_dir = os.path.join(AUDIO_AI_OPTIMIZED_DIR, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Load the audio file
        try:
            audio = AudioSegment.from_file(file_path)
            original_format = {
                'channels': audio.channels,
                'frame_rate': audio.frame_rate,
                'sample_width': audio.sample_width,
                'duration_seconds': len(audio) / 1000
            }
            logger.info(f"Audio loaded successfully - Duration: {original_format['duration_seconds']:.2f}s, "
                       f"Channels: {original_format['channels']}, "
                       f"Sample rate: {original_format['frame_rate']}Hz, "
                       f"Bit depth: {original_format['sample_width']*8} bits")
            
            # If max_duration is specified, trim the audio
            if max_duration is not None and original_format['duration_seconds'] > max_duration:
                logger.info(f"Trimming audio to first {max_duration} seconds")
                audio = audio[:max_duration * 1000]  # Convert seconds to milliseconds
                processed_duration = max_duration
            else:
                processed_duration = original_format['duration_seconds']
            
            # Standardize to mono 16kHz 16-bit
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            logger.info(f"Audio standardized to mono 16kHz 16-bit PCM")
        except Exception as e:
            logger.error(f"Failed to load audio file {os.path.basename(file_path)}: {str(e)}")
            return False, f"Error: Failed to load audio file - {str(e)}"
        
        # Get the base filename without extension (for metadata only)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create a metadata file for this optimization run
        metadata = {
            'original_file': os.path.basename(file_path),
            'original_format': original_format,
            'processed_duration': processed_duration,
            'max_duration_applied': max_duration is not None,
            'timestamp': timestamp,
            'chunk_lengths': chunk_lengths,
            'overlap_lengths': overlap_lengths,
            'combinations': [],
            'skipped_combinations': []
        }
        
        # Calculate valid combinations (where overlap < chunk length)
        valid_combinations = []
        skipped_combinations = []
        
        for chunk_length in sorted(chunk_lengths):
            for overlap_length in sorted(overlap_lengths):
                # Skip invalid combinations where overlap >= chunk length
                if overlap_length >= chunk_length:
                    skipped_combinations.append({
                        'chunk_length': chunk_length,
                        'overlap_length': overlap_length,
                        'reason': "Overlap length must be less than chunk length"
                    })
                    continue
                
                valid_combinations.append((chunk_length, overlap_length))
        
        total_combinations = len(valid_combinations)
        processed_combinations = 0
        total_files_created = 0
        
        logger.info(f"Processing {total_combinations} valid combinations "
                   f"(skipped {len(skipped_combinations)} invalid combinations)")
        
        # Process each valid combination of chunk length and overlap length
        for chunk_length, overlap_length in valid_combinations:
            processed_combinations += 1
            
            # Create a subdirectory for this combination
            combo_dir = os.path.join(run_dir, f"chunk{chunk_length}s_overlap{overlap_length}s")
            os.makedirs(combo_dir, exist_ok=True)
            
            logger.info(f"=== COMBINATION {processed_combinations}/{total_combinations} ===")
            logger.info(f"Chunk length: {chunk_length}s, Overlap length: {overlap_length}s")
            
            try:
                # Calculate chunk parameters in milliseconds
                chunk_length_ms = chunk_length * 1000
                overlap_ms = overlap_length * 1000
                
                # Calculate the number of chunks
                audio_length_ms = len(audio)
                
                # When using overlap, each chunk (except the first) starts (chunk_length_ms - overlap_ms) after the previous one
                effective_chunk_length = chunk_length_ms - overlap_ms
                total_chunks = max(1, int((audio_length_ms - overlap_ms) / effective_chunk_length) + 1)
                
                logger.info(f"Audio length: {audio_length_ms/1000:.2f}s, "
                           f"Calculated {total_chunks} chunks")
                
                # Track files created for this combination
                combo_files = []
                
                # Process each chunk
                for i in range(total_chunks):
                    # Calculate the start and end times for this chunk
                    # With overlap, each chunk starts (chunk_length_ms - overlap_ms) after the previous one
                    chunk_start = i * effective_chunk_length
                    
                    # Ensure we don't go beyond the end of the audio
                    chunk_end = min(chunk_start + chunk_length_ms, audio_length_ms)
                    
                    # Calculate padding before and after
                    padding_before_ms = overlap_ms if i > 0 else 0
                    padding_after_ms = overlap_ms if i < total_chunks - 1 else 0
                    
                    # Calculate actual start and end positions including padding
                    actual_start = max(0, chunk_start - padding_before_ms)
                    actual_end = min(audio_length_ms, chunk_end + padding_after_ms)
                    
                    # Extract the chunk with padding
                    padded_chunk = audio[actual_start:actual_end]
                    
                    # Log detailed information about this chunk
                    logger.debug(f"Chunk {i+1}/{total_chunks}: " +
                                f"Base: {chunk_start/1000:.2f}s-{chunk_end/1000:.2f}s, " +
                                f"With padding: {actual_start/1000:.2f}s-{actual_end/1000:.2f}s " +
                                f"(Padding before: {padding_before_ms/1000:.2f}s, after: {padding_after_ms/1000:.2f}s)")
                    
                    # Create a filename for this chunk
                    chunk_filename = f"ai_audio_chunk{chunk_length}s_overlap{overlap_length}s_{i+1:03d}.wav"
                    chunk_path = os.path.join(combo_dir, chunk_filename)
                    
                    # Export the chunk
                    padded_chunk.export(chunk_path, format="wav")
                    
                    # Add this file to the list
                    combo_files.append({
                        'filename': chunk_filename,
                        'chunk_number': i + 1,
                        'base_start_time': chunk_start / 1000,  # Convert to seconds
                        'base_end_time': chunk_end / 1000,
                        'actual_start_time': actual_start / 1000,
                        'actual_end_time': actual_end / 1000,
                        'duration': len(padded_chunk) / 1000,
                        'has_padding_before': padding_before_ms > 0,
                        'has_padding_after': padding_after_ms > 0,
                        'padding_before': padding_before_ms / 1000,
                        'padding_after': padding_after_ms / 1000
                    })
                    
                    total_files_created += 1
                
                # Add this combination to the metadata
                metadata['combinations'].append({
                    'chunk_length': chunk_length,
                    'overlap_length': overlap_length,
                    'total_chunks': total_chunks,
                    'directory': os.path.basename(combo_dir),
                    'files': combo_files
                })
                
                logger.info(f"Successfully created {len(combo_files)} chunk files for combination: chunk{chunk_length}s_overlap{overlap_length}s")
                
            except Exception as e:
                error_msg = f"Error processing combination (chunk: {chunk_length}s, overlap: {overlap_length}s): {str(e)}"
                logger.error(error_msg)
                metadata['skipped_combinations'].append({
                    'chunk_length': chunk_length,
                    'overlap_length': overlap_length,
                    'reason': str(e)
                })
                # Continue with the next combination
        
        # Add skipped combinations to metadata
        for combo in skipped_combinations:
            metadata['skipped_combinations'].append(combo)
        
        # Save the metadata
        metadata_path = os.path.join(run_dir, "optimization_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate a summary
        summary = (
            f"Audio optimization completed for {os.path.basename(file_path)}\n\n"
            f"- Original duration: {original_format['duration_seconds']:.2f} seconds\n"
            f"- Processed duration: {processed_duration:.2f} seconds"
        )
        
        if max_duration is not None and original_format['duration_seconds'] > max_duration:
            summary += f" (limited to first {max_duration} seconds)\n"
        else:
            summary += " (full audio)\n"
            
        summary += (
            f"- Processed {len(metadata['combinations'])} valid combinations\n"
            f"- Created {total_files_created} individual audio files\n"
            f"- Skipped {len(metadata['skipped_combinations'])} invalid combinations\n"
            f"- Output directory: {run_dir}\n\n"
        )
        
        # Add details for each combination
        summary += "Processed combinations:\n"
        for combo in metadata['combinations']:
            chunk_length = combo['chunk_length']
            overlap_length = combo['overlap_length']
            total_chunks = combo['total_chunks']
            
            # Calculate the maximum possible duration with padding
            max_chunk_duration = chunk_length
            if overlap_length > 0:
                max_chunk_duration += (overlap_length * 2)  # Add padding before and after
                
            summary += (
                f"- Chunk: {chunk_length}s, Overlap: {overlap_length}s - {total_chunks} chunks\n"
                f"  (Base chunk: {chunk_length}s, Middle chunks with padding: up to {max_chunk_duration}s)\n"
            )
        
        if metadata['skipped_combinations']:
            summary += "\nSkipped combinations:\n"
            for combo in metadata['skipped_combinations']:
                summary += f"- Chunk: {combo['chunk_length']}s, Overlap: {combo['overlap_length']}s - Reason: {combo['reason']}\n"
        
        # Add explanation of padding approach
        summary += "\nPadding Approach:\n"
        summary += "- First chunk: No padding before, may have padding after\n"
        summary += "- Middle chunks: Padding before and after (if overlap > 0)\n"
        summary += "- Last chunk: May have padding before, no padding after\n"
        summary += "- Example: For 30s chunks with 5s overlap, middle chunks will be 40s (5s + 30s + 5s)\n"
        
        return True, summary 