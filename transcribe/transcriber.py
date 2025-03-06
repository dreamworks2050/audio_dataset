import os
import subprocess
import tempfile
from pydub import AudioSegment
from utils.logger import logger
import time

class TranscriptionService:
    """Service for transcribing audio using whisper.cpp."""
    
    def __init__(self, model_path):
        """Initialize the transcription service.
        
        Args:
            model_path (str): Path to the whisper.cpp model file
        """
        self.model_path = model_path
        self.whisper_cpp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "whisper.cpp")
        self.whisper_bin = os.path.join(self.whisper_cpp_path, "build", "bin", "whisper-cli")
        
        # Verify model and binary exist
        self._verify_dependencies()
    
    def _verify_dependencies(self):
        """Verify that the required dependencies exist."""
        # Check if whisper.cpp binary exists
        if not os.path.exists(self.whisper_bin):
            logger.error(f"whisper.cpp binary not found at {self.whisper_bin}")
            raise FileNotFoundError(f"whisper.cpp binary not found at {self.whisper_bin}")
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.error(f"Whisper model not found at {self.model_path}")
            raise FileNotFoundError(f"Whisper model not found at {self.model_path}")
        
        logger.info(f"Transcription service initialized with model: {os.path.basename(self.model_path)}")
    
    def transcribe_chunk(self, audio_chunk, lang_code="en", initial_prompt=None):
        """Transcribe an audio chunk using whisper.cpp.
        
        Args:
            audio_chunk (AudioSegment): The audio chunk to transcribe
            lang_code (str, optional): Language code for transcription. Defaults to "en".
            initial_prompt (str, optional): Initial prompt for the transcription. Defaults to None.
            
        Returns:
            str: The transcribed text or None if transcription failed
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
            try:
                # Export audio chunk to temporary file
                logger.debug(f"Exporting audio chunk to temporary file: {temp_filename}")
                audio_chunk.export(temp_filename, format="wav")
                
                # Check audio properties
                logger.debug(f"Audio chunk properties: duration={len(audio_chunk)/1000:.2f}s, dBFS={audio_chunk.dBFS:.2f}")
                if audio_chunk.dBFS < -50:
                    logger.warning(f"Audio chunk is too quiet (dBFS: {audio_chunk.dBFS:.2f}), may result in poor transcription")
                
                # Build command
                cmd = [
                    self.whisper_bin,
                    "--flash-attn",
                    "--threads", "8",
                    "-m", self.model_path,
                    "-f", temp_filename,
                    "-l", lang_code,
                    "--output-txt"
                ]
                
                # Log language code specifically to debug any issues
                logger.warning(f"WHISPER CMD LANGUAGE: Lang_code='{lang_code}' for transcription")
                logger.debug(f"Using language code: '{lang_code}' for transcription")
                
                # Add initial prompt if provided
                if initial_prompt:
                    try:
                        # Log original prompt encoding for debugging
                        prompt_bytes = initial_prompt.encode('utf-8')
                        logger.debug(f"Original prompt encoding check - bytes: {len(prompt_bytes)}")
                        
                        # Sanitize the prompt: remove any characters that could cause issues with command line
                        sanitized_prompt = initial_prompt.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
                        
                        # Add the prompt to the command arguments list
                        # The subprocess.run with a list of arguments handles quoting automatically
                        cmd.extend(["--prompt", sanitized_prompt])
                        
                        # Log the full prompt for debugging
                        logger.debug(f"Using prompt (estimated tokens: {len(sanitized_prompt) // 4}): {sanitized_prompt}")
                        # Also log the token count for easier debugging
                        logger.debug(f"Prompt token count (estimated): {len(sanitized_prompt) // 4}")
                    except UnicodeEncodeError as e:
                        # If we have encoding issues with the prompt, log and continue without it
                        logger.warning(f"Unicode encoding issue with prompt: {str(e)}")
                        logger.warning("Continuing transcription without the prompt")
                
                # Run whisper.cpp
                cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
                logger.debug(f"Running whisper.cpp command: {cmd_str}")
                
                try:
                    # Add a timeout to prevent the process from hanging indefinitely
                    start_time = time.time()
                    
                    # Use subprocess with environment set to ensure proper encoding
                    result = subprocess.run(
                        cmd,  # Using the list form ensures proper argument handling
                        capture_output=True,
                        text=False,  # Changed to False to get binary data
                        check=False,
                        timeout=300,  # 5 minute timeout for each chunk
                        env=dict(os.environ, PYTHONIOENCODING='utf-8')  # Ensure Python uses UTF-8 for I/O
                    )
                    
                    elapsed_time = time.time() - start_time
                    logger.debug(f"Whisper.cpp process completed in {elapsed_time:.2f} seconds")
                
                except subprocess.TimeoutExpired:
                    logger.error(f"Whisper.cpp process timed out after 300 seconds for file: {temp_filename}")
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error running whisper.cpp: {str(e)}")
                    return None
                
                # Check for errors - decode stderr with error handling
                stderr_text = ""
                try:
                    if result.stderr:
                        stderr_text = result.stderr.decode('utf-8', errors='replace')
                except Exception as e:
                    stderr_text = f"[Error decoding stderr: {str(e)}]"
                
                stdout_text = ""
                try:
                    if result.stdout:
                        stdout_text = result.stdout.decode('utf-8', errors='replace')
                except Exception as e:
                    stdout_text = f"[Error decoding stdout: {str(e)}]"
                
                if result.returncode != 0:
                    logger.error(f"whisper.cpp failed with return code {result.returncode}")
                    logger.error(f"Error details: {stderr_text}")
                    logger.debug(f"Command output: {stdout_text}")
                    return None
                
                # Get transcription from output file
                txt_file = temp_filename + ".txt"
                if os.path.exists(txt_file):
                    try:
                        # First try with strict UTF-8 decoding
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            transcription = f.read().strip()
                    except UnicodeDecodeError as ude:
                        logger.warning(f"UTF-8 decoding error in output file: {str(ude)}")
                        
                        # Check file size and content type for debugging
                        file_size = os.path.getsize(txt_file)
                        logger.debug(f"Output file size: {file_size} bytes")
                        
                        try:
                            # Try to detect encoding - check first few bytes
                            with open(txt_file, 'rb') as f:
                                header_bytes = f.read(min(file_size, 16))
                                byte_str = ' '.join(f'{b:02x}' for b in header_bytes)
                                logger.debug(f"First bytes: {byte_str}")
                            
                            # Enhanced handling for Korean and other East Asian languages
                            # Try common East Asian encodings if the file appears to contain East Asian text
                            for encoding in ['utf-8', 'cp949', 'euc-kr', 'euc-jp', 'shift-jis', 'gb2312', 'gbk']:
                                try:
                                    with open(txt_file, 'r', encoding=encoding) as f:
                                        candidate_text = f.read().strip()
                                        # Check if this looks like valid text (non-empty and not just control chars)
                                        if candidate_text and any(c.isalpha() for c in candidate_text):
                                            transcription = candidate_text
                                            logger.info(f"Successfully decoded file using {encoding} encoding")
                                            break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                # If none of the specific encodings worked, fall back to UTF-8 with replacement
                                logger.warning("All specific encoding attempts failed, using UTF-8 with replacement")
                                with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                                    transcription = f.read().strip()
                                logger.info(f"Successfully read file with UTF-8 replacement for invalid characters")
                        
                        except Exception as e2:
                            logger.error(f"Enhanced error handling failed: {str(e2)}")
                            # Last resort: read as binary and decode with explicit error handling
                            try:
                                with open(txt_file, 'rb') as f:
                                    binary_data = f.read()
                                transcription = binary_data.decode('utf-8', errors='replace').strip()
                                logger.info(f"Successfully decoded file using binary read with replacement")
                            except Exception as e3:
                                logger.error(f"All methods to read the output file failed: {str(e3)}")
                                return None
                    
                    logger.debug(f"Transcription result: {transcription[:100]}...")
                    os.remove(txt_file)
                    return transcription
                else:
                    logger.warning(f"No output file found at {txt_file}")
                    logger.debug(f"Command stdout: {stdout_text}")
                    logger.debug(f"Command stderr: {stderr_text}")
                    return None
                
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
                return None
            finally:
                # Clean up temporary files
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    logger.debug(f"Removed temporary audio file: {temp_filename}") 