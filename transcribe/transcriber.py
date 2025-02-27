import os
import subprocess
import tempfile
from pydub import AudioSegment
from utils.logger import logger

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
                
                # Add initial prompt if provided
                if initial_prompt:
                    # Use --prompt directly instead of --prompt-file
                    cmd.extend(["--prompt", initial_prompt])
                    logger.debug(f"Using prompt: {initial_prompt[:50]}...")
                
                # Run whisper.cpp
                cmd_str = ' '.join(cmd)
                logger.debug(f"Running whisper.cpp command: {cmd_str}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                # Check for errors
                if result.returncode != 0:
                    logger.error(f"whisper.cpp failed with return code {result.returncode}: {result.stderr}")
                    logger.debug(f"Command output: {result.stdout}")
                    return None
                
                # Get transcription from output file
                txt_file = temp_filename + ".txt"
                if os.path.exists(txt_file):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcription = f.read().strip()
                    logger.debug(f"Transcription result: {transcription[:100]}...")
                    os.remove(txt_file)
                    return transcription
                else:
                    logger.warning(f"No output file found at {txt_file}")
                    logger.debug(f"Command stdout: {result.stdout}")
                    logger.debug(f"Command stderr: {result.stderr}")
                    return None
                
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}", exc_info=True)
                return None
            finally:
                # Clean up temporary files
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    logger.debug(f"Removed temporary audio file: {temp_filename}") 