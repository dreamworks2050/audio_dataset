import os
import subprocess
import threading
import time
import sqlite3
from typing import Optional
import gradio as gr
from pydub import AudioSegment
import yt_dlp
from datetime import datetime, timedelta, timezone
import ollama
from pydantic import BaseModel
import json
import shutil
from logging_config import setup_logger
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import re

logger = setup_logger()

LANGUAGE_MAP = {
    "Korean": "ko",
    "English": "en",
    "Chinese": "zh",
    "Vietnamese": "vi",
    "Spanish": "es"
}

def seconds_to_timecode(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"[{hours:04d}:{minutes:02d}:{secs:02d}]"

class CleanedTranscription(BaseModel):
    cleaned_text: str

class AudioChunker:
    def __init__(self, chunk_size=60000, padding=10000, min_silence_len=250, silence_thresh=-35):
        self.buffer = AudioSegment.empty()
        self.chunk_size = chunk_size
        self.padding = padding
        self.total_chunk_size = chunk_size + 2 * padding
        self.frame_rate = 16000
        self.sample_width = 2
        self.channels = 1
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh

    def add_audio(self, raw_data):
        new_segment = AudioSegment(
            raw_data,
            sample_width=self.sample_width,
            frame_rate=self.frame_rate,
            channels=self.channels
        )
        self.buffer += new_segment
        while len(self.buffer) >= self.total_chunk_size:
            chunk = self.buffer[:self.total_chunk_size]
            yield chunk
            self.buffer = self.buffer[self.chunk_size:]

def get_metadata_start_time(info):
    try:
        if info.get('live_status') == 'is_live':
            if info.get('start_time'):
                return datetime.fromtimestamp(info['start_time'], timezone.utc)
            elif info.get('release_timestamp'):
                return datetime.fromtimestamp(info['release_timestamp'], timezone.utc)
            elif info.get('release_date'):
                return datetime.fromisoformat(info['release_date'].replace('Z', '+00:00'))
    except Exception as e:
        logger.warning(f"Failed to extract metadata start time: {e}")
    return None

class TranscriptionSystem:
    def __init__(self, model_path: str, show_ui_update_logs: bool = True):
        logger.debug("Initializing TranscriptionSystem")
        self.model_path = model_path
        self.db_path = 'transcriptions.db'
        self.show_ui_update_logs = show_ui_update_logs
        self.setup_database()
        self.chunker = AudioChunker(chunk_size=60000, padding=10000)
        self.livestream_start_time = None
        self.stream_start_time = None
        self.first_chunk_time = None
        self.transcriptions = []
        self.db_lock = threading.Lock()
        self.chunk_count = 0
        self.save_texts_to_files = True
        self.is_running = False
        self.process = None
        
        # Initialize the transcription service
        try:
            from transcribe.transcriber import TranscriptionService
            self.transcription_service = TranscriptionService(model_path)
            logger.info("Transcription service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize transcription service: {e}")
            raise
        
        self.first_run_system_prompt = "You are an AI tasked with cleaning transcriptions from a livestream. Return the response as a JSON object with a single key 'cleaned_text' containing the cleaned transcription."
        self.first_run_task_prompt = "Clean the transcription of Chunk 1 using Chunk 2 as forward context: Chunk 1: {chunk1_text}, Chunk 2: {chunk2_text}"
        self.subsequent_system_prompt = "You are an AI tasked with cleaning transcriptions from a livestream. Return the response as a JSON object with a single key 'cleaned_text' containing the cleaned transcription."
        self.subsequent_task_prompt = "Clean the current chunk as a continuation: Previous: {prev_cleaned_text}, Current: {raw_text}, Next: {next_raw_text}"
        
        prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.json')
        logger.debug(f"Attempting to load prompts from: {prompt_path}")
        
        if not os.path.exists(prompt_path):
            logger.critical(f"⚠️ 🔴 CRITICAL ERROR: prompt.json not found at {prompt_path}! 🔴 ⚠️")
            print(f"\n\033[91m⚠️ 🔴 CRITICAL ERROR: prompt.json not found at {prompt_path}! 🔴 ⚠️\033[0m")
        else:
            try:
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                    required_keys = [
                        'first_run_system_prompt', 'first_run_task_prompt',
                        'subsequent_system_prompt', 'subsequent_task_prompt'
                    ]
                    missing_keys = [key for key in required_keys if key not in prompts]
                    if missing_keys:
                        error_msg = f"Missing required keys in prompt.json: {', '.join(missing_keys)}"
                        logger.critical(f"⚠️ 🔴 CRITICAL ERROR: {error_msg} 🔴 ⚠️")
                        print(f"\n\033[91m⚠️ 🔴 CRITICAL ERROR: {error_msg} 🔴 ⚠️\033[0m")
                        raise ValueError(error_msg)
                        
                    prompts = self.validate_and_correct_prompts(prompts)
                    self.first_run_system_prompt = prompts['first_run_system_prompt']
                    self.first_run_task_prompt = prompts['first_run_task_prompt']
                    self.subsequent_system_prompt = prompts['subsequent_system_prompt']
                    self.subsequent_task_prompt = prompts['subsequent_task_prompt']
                    logger.info("✅ Prompts successfully loaded from prompt.json")
                    print("\n\033[92m✅ Successfully loaded all prompts from prompt.json!\033[0m")
            except FileNotFoundError:
                logger.critical("⚠️ 🔴 CRITICAL ERROR: prompt.json NOT FOUND! The system will use basic defaults but results will be SIGNIFICANTLY DEGRADED! 🔴 ⚠️")
                logger.critical("⚠️ 🔴 IMPORTANT: You MUST create a proper prompt.json file for optimal performance! 🔴 ⚠️")
            except json.JSONDecodeError as e:
                logger.critical(f"⚠️ 🔴 CRITICAL ERROR: Failed to parse prompt.json: {e}")
                logger.critical("⚠️ 🔴 IMPORTANT: The prompt.json file contains invalid JSON! Results will be SIGNIFICANTLY DEGRADED! 🔴 ⚠️")
                print(f"\n\033[91m⚠️ 🔴 CRITICAL ERROR: Invalid JSON in prompt.json! Fix immediately for optimal results! 🔴 ⚠️\033[0m")
            except ValueError as e:
                logger.critical(f"⚠️ 🔴 CRITICAL ERROR: Invalid prompt.json format: {e}")
                logger.critical("⚠️ 🔴 IMPORTANT: The prompt.json file is missing required keys! Results will be SIGNIFICANTLY DEGRADED! 🔴 ⚠️")
                print(f"\n\033[91m⚠️ 🔴 CRITICAL ERROR: prompt.json is missing required keys! Fix immediately! 🔴 ⚠️\033[0m")

        try:
            whisper_prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'whisper_prompt.json')
            logger.debug(f"Loading whisper prompts from: {whisper_prompt_path}")
            
            if os.path.exists(whisper_prompt_path):
                with open(whisper_prompt_path, 'r', encoding='utf-8') as f:
                    self.whisper_prompts = json.load(f)
                logger.info(f"Whisper prompts loaded from {whisper_prompt_path}")
            else:
                logger.warning(f"whisper_prompt.json not found at {whisper_prompt_path}; using default prompts")
                self.whisper_prompts = {
                    "en": "Continuation: ",
                    "ko": "계속: ",
                    "zh": "继续: ",
                    "vi": "Tiếp tục: ",
                    "es": "Continuación: "
                }
        except FileNotFoundError:
            logger.warning("whisper_prompt.json not found; using default prompts")
            self.whisper_prompts = {
                "en": "Continuation: ",
                "ko": "계속: ",
                "zh": "继续: ",
                "vi": "Tiếp tục: ",
                "es": "Continuación: "
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse whisper_prompt.json: {e}")
            logger.warning("Using default whisper prompts due to JSON error")
            self.whisper_prompts = {
                "en": "Continuation: ",
                "ko": "계속: ",
                "zh": "继续: ",
                "vi": "Tiếp tục: ",
                "es": "Continuación: "
            }

    def validate_and_correct_prompts(self, prompts):
        """Validate placeholders and correct unescaped JSON-like structures in prompts."""
        placeholder_pattern = r'(?<!{){([^{}]+)}(?!})'
        json_like_pattern = r'(?<!{){\s*\'[^\'\n]+\'\s*:\s*\'[^\'\n]+\'\s*}(?!})'
        
        task_prompts = {
            'first_run_task_prompt': {'chunk1_text', 'chunk2_text'},
            'subsequent_task_prompt': {'prev_cleaned_text', 'prev_raw_text', 'raw_text', 'next_raw_text'}
        }
        
        corrected_prompts = {}
        
        for key in prompts:
            original_prompt = prompts[key]
            corrected_prompt = original_prompt
            
            placeholders = set(re.findall(placeholder_pattern, original_prompt))
            json_match_list = list(re.finditer(json_like_pattern, original_prompt))
            json_placeholders = set(match.group(0)[1:-1] for match in json_match_list)
            
            non_json_placeholders = placeholders - json_placeholders
            
            if key in ['first_run_system_prompt', 'subsequent_system_prompt']:
                if non_json_placeholders:
                    logger.warning(f"System prompt '{key}' contains unexpected placeholders: {non_json_placeholders}. Ensure curly braces are escaped ({{}}) if intended as literal text.")
            
            elif key in task_prompts:
                expected_vars = task_prompts[key]
                extra_vars = non_json_placeholders - expected_vars
                missing_vars = expected_vars - non_json_placeholders
                if extra_vars:
                    logger.warning(f"Task prompt '{key}' contains unexpected placeholders: {extra_vars}. Ensure curly braces are escaped ({{}}) if intended as literal text.")
                if missing_vars:
                    logger.warning(f"Task prompt '{key}' is missing expected placeholders: {missing_vars}. Required placeholders: {expected_vars}")
            
            if json_match_list:
                for match in json_match_list:
                    unescaped = match.group(0)
                    escaped = '{{' + unescaped[1:-1] + '}}'
                    if unescaped != escaped:
                        logger.warning(f"Prompt '{key}' contains unescaped JSON-like structure '{unescaped}'. Correcting to '{escaped}' on-the-fly.")
                        corrected_prompt = corrected_prompt.replace(unescaped, escaped)
            
            corrected_prompts[key] = corrected_prompt
        
        return corrected_prompts

    def setup_database(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT,
                end_time TEXT,
                text TEXT,
                cleaned_text TEXT DEFAULT '',
                timestamp TEXT,
                processing_state TEXT DEFAULT 'raw'
            )
        ''')
        cursor = self.conn.execute("PRAGMA table_info(transcriptions)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'cleaned_text' not in columns:
            self.conn.execute('ALTER TABLE transcriptions ADD COLUMN cleaned_text TEXT DEFAULT ""')
        if 'processing_state' not in columns:
            self.conn.execute('ALTER TABLE transcriptions ADD COLUMN processing_state TEXT DEFAULT "raw"')
        self.conn.commit()
        logger.debug("Database setup complete")

    def reset_database(self):
        try:
            if hasattr(self, 'conn') and self.conn:
                try:
                    self.conn.close()
                    logger.debug("Database connection closed")
                except sqlite3.Error as e:
                    logger.warning(f"Error closing database connection: {e}")
                    
            if os.path.exists(self.db_path):
                try:
                    os.remove(self.db_path)
                    logger.info(f"Deleted database file: {self.db_path}")
                except Exception as e:
                    logger.error(f"Failed to delete database file {self.db_path}: {e}")
                    raise
            
            self.setup_database()
            self.chunk_count = 0
            self.transcriptions = []
            logger.info("Database recreated from scratch")
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise

    def get_stream_url(self, youtube_url: str) -> str:
        logger.debug(f"Fetching stream URL: {youtube_url}")
        ydl_opts = {'format': 'bestaudio', 'quiet': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                self.livestream_start_time = get_metadata_start_time(info)
                if self.livestream_start_time:
                    logger.info(f"Livestream started at: {self.livestream_start_time}")
                if 'url' not in info:
                    logger.error(f"No URL found in extracted info for {youtube_url}")
                    raise ValueError(f"Failed to extract audio URL from {youtube_url}")
                return info['url']
        except Exception as e:
            logger.error(f"Failed to extract info from YouTube URL {youtube_url}: {e}")
            raise

    def process_stream(self, youtube_url: str, save_chunks: bool, empty_db: bool, empty_audio_folder: bool, language: str):
        logger.debug("Starting stream processing")
        # These settings are now handled in start_transcription, so we only process the stream here
        stream_url = self.get_stream_url(youtube_url)
        self.stream_start_time = datetime.now(timezone.utc)
        if not self.livestream_start_time:
            self.livestream_start_time = self.stream_start_time

        ffmpeg_cmd = [
            'ffmpeg', '-i', stream_url,
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-f', 'wav', '-loglevel', 'quiet', 'pipe:1'
        ]
        self.process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=16000
        )
        logger.info("FFmpeg started")
        self.is_running = True

        chunk_count = 0
        previous_raw_text = ""
        try:
            while self.is_running:
                raw_data = self.process.stdout.read(16000)
                if not raw_data:
                    logger.info("Stream ended")
                    break
                for chunk in self.chunker.add_audio(raw_data):
                    if not self.is_running:
                        break
                    chunk_count += 1
                    self.chunk_count = chunk_count
                    chunk_start_offset = (chunk_count - 1) * self.chunker.chunk_size / 1000.0
                    chunk_start_time = self.stream_start_time + timedelta(
                        seconds=chunk_start_offset - self.chunker.padding / 1000.0
                    )
                    chunk_end_time = chunk_start_time + timedelta(
                        seconds=self.chunker.total_chunk_size / 1000.0
                    )

                    start_timecode_seconds = (chunk_start_time - self.livestream_start_time).total_seconds()
                    end_timecode_seconds = (chunk_end_time - self.livestream_start_time).total_seconds()
                    start_timecode = seconds_to_timecode(start_timecode_seconds)
                    end_timecode = seconds_to_timecode(end_timecode_seconds)

                    if not self.first_chunk_time:
                        self.first_chunk_time = chunk_start_time
                    logger.info(f"Chunk {chunk_count} - Transcription covers: {start_timecode} to {end_timecode}")
                    if save_chunks:
                        chunk_filename = f"audio_chunks/chunk_{chunk_count}.wav"
                        chunk.export(chunk_filename, format="wav")
                        logger.debug(f"Saved chunk: {chunk_filename}")

                    lang_code = LANGUAGE_MAP.get(language, "en")
                    if not hasattr(self, 'whisper_prompts') or not self.whisper_prompts:
                        whisper_prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'whisper_prompt.json')
                        try:
                            if os.path.exists(whisper_prompt_path):
                                with open(whisper_prompt_path, 'r', encoding='utf-8') as f:
                                    self.whisper_prompts = json.load(f)
                                logger.info(f"Whisper prompts loaded from {whisper_prompt_path} during transcription")
                            else:
                                self.whisper_prompts = {
                                    "en": "Continuation: ",
                                    "ko": "계속: ",
                                    "zh": "继续: ",
                                    "vi": "Tiếp tục: ",
                                    "es": "Continuación: "
                                }
                                logger.warning("Whisper prompts file not found; using hardcoded defaults")
                        except Exception as e:
                            self.whisper_prompts = {
                                "en": "Continuation: ",
                                "ko": "계속: ",
                                "zh": "继续: ",
                                "vi": "Tiếp tục: ",
                                "es": "Continuación: "
                            }
                            logger.warning(f"Error loading whisper_prompt.json: {e}, using defaults")
                    
                    prefix = self.whisper_prompts.get(lang_code, "")
                    if previous_raw_text:
                        trimmed_prev = " ".join(previous_raw_text.split()[-30:])
                        initial_prompt = f"{prefix}{trimmed_prev}"
                    else:
                        initial_prompt = ""
                    transcription = self.transcribe_chunk(chunk, chunk_count, language, initial_prompt)

                    prompt_snippet = ' '.join(initial_prompt.split()[:5]) + '...' if initial_prompt else 'No prompt'
                    if transcription:
                        logger.info(f"Transcription successful for chunk {chunk_count} with prompt: {prompt_snippet}")
                        self.store_transcription(start_timecode, end_timecode, transcription['text'])
                        self.transcriptions.append((
                            start_timecode_seconds,
                            end_timecode_seconds,
                            transcription['text']
                        ))
                        previous_raw_text = transcription['text']
                    else:
                        logger.warning(f"Transcription failed for chunk {chunk_count} with prompt: {prompt_snippet}")
                        previous_raw_text = ""

        finally:
            if self.process:
                self.process.terminate()
            self.is_running = False
            logger.info("Transcription process stopped")

    def stop_process(self):
        """Stop the transcription process without resetting settings."""
        if self.is_running and self.process:
            self.is_running = False
            self.process.terminate()
            logger.info("Transcription process terminated via stop command")
            self.process = None

    def transcribe_chunk(self, audio_chunk: AudioSegment, chunk_number: int, language: str, initial_prompt: str = "") -> Optional[dict]:
        logger.debug(f"Transcribing chunk {chunk_number} in {language}")
        temp_file = f"temp_chunk_{chunk_number}.wav"
        try:
            audio_chunk.export(temp_file, format="wav")
            dbfs = audio_chunk.dBFS
            if dbfs < -50:
                logger.debug(f"Chunk {chunk_number} too quiet (dBFS: {dbfs})")
                return None
            lang_code = LANGUAGE_MAP.get(language, "en")
            transcription = self.transcription_service.transcribe_chunk(
                audio_chunk,
                lang_code=lang_code,
                initial_prompt=initial_prompt if initial_prompt else None
            )
            if not transcription:
                logger.debug(f"No transcription for chunk {chunk_number} in {language}")
                return None
            logger.debug(f"Transcription successful for chunk {chunk_number} in {language}")
            return {"text": transcription}
            if result.returncode != 0:
                logger.error(f"Whisper failed for chunk {chunk_number} with return code {result.returncode}: {result.stderr}")
                return None
            raw_output = result.stdout.strip()
            if not raw_output:
                logger.debug(f"No transcription for chunk {chunk_number} in {language}. Whisper stderr: {result.stderr}")
                return None
            logger.debug(f"Transcription successful for chunk {chunk_number} in {language}")
            return {"text": raw_output}
        except Exception as e:
            logger.error(f"Unexpected error in transcription for chunk {chunk_number}: {e}")
            return None
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def store_transcription(self, start_timecode: str, end_timecode: str, text: str):
        with self.db_lock:
            cursor = self.conn.execute(
                'INSERT INTO transcriptions (start_time, end_time, text, cleaned_text, timestamp, processing_state) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (start_timecode, end_timecode, text, '', start_timecode, 'raw')
            )
            chunk_id = cursor.lastrowid
            self.conn.commit()
            raw_text = f"{start_timecode} [{start_timecode}-{end_timecode}] {text}"
            self.save_transcriptions_to_files(raw_text=raw_text)

        logger.debug(f"Stored transcription for {start_timecode}-{end_timecode} with ID {chunk_id}, chunk_count={self.chunk_count}")
        
        if self.chunk_count >= 2:
            with self.db_lock:
                cursor = self.conn.execute(
                    'SELECT id FROM transcriptions WHERE id < ? ORDER BY id DESC LIMIT 1',
                    (chunk_id,)
                )
                prev_row = cursor.fetchone()
                if prev_row:
                    prev_chunk_id = prev_row[0]
                    logger.debug(f"Triggering AI cleaning for chunk {prev_chunk_id} after storing chunk {chunk_id} (chunk_count={self.chunk_count})")
                    clean_thread = threading.Thread(
                        target=self.clean_chunk, 
                        args=(prev_chunk_id,),
                        daemon=True
                    )
                    clean_thread.start()
                else:
                    logger.warning(f"No previous chunk found for chunk {chunk_id}")

    def clean_chunk(self, chunk_id: int):
        prompt_attrs_check = []
        if not hasattr(self, 'first_run_system_prompt') or not self.first_run_system_prompt:
            prompt_attrs_check.append('first_run_system_prompt')
            
        if not hasattr(self, 'first_run_task_prompt') or not self.first_run_task_prompt:
            prompt_attrs_check.append('first_run_task_prompt')
            
        if not hasattr(self, 'subsequent_system_prompt') or not self.subsequent_system_prompt:
            prompt_attrs_check.append('subsequent_system_prompt')
            
        if not hasattr(self, 'subsequent_task_prompt') or not self.subsequent_task_prompt:
            prompt_attrs_check.append('subsequent_task_prompt')
            
        if prompt_attrs_check:
            for attr in prompt_attrs_check:
                logger.critical(f"⚠️ 🔴 CRITICAL ERROR: Prompt attribute '{attr}' missing in clean_chunk! 🔴 ⚠️")
            logger.critical("⚠️ 🔴 EMERGENCY: Using fallback prompts with SEVERELY DEGRADED PERFORMANCE! 🔴 ⚠️")
            print("\n\033[91m⚠️ 🔴 CRITICAL: Missing prompt attributes while cleaning! Results will be poor! 🔴 ⚠️\033[0m")
            print("\n\033[91mAttempting to load prompt.json again as emergency measure...\033[0m")
            
            prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.json')
            if os.path.exists(prompt_path):
                try:
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        prompts = json.load(f)
                        if all(k in prompts for k in ['first_run_system_prompt', 'first_run_task_prompt', 
                                                     'subsequent_system_prompt', 'subsequent_task_prompt']):
                            self.first_run_system_prompt = prompts['first_run_system_prompt']
                            self.first_run_task_prompt = prompts['first_run_task_prompt']
                            self.subsequent_system_prompt = prompts['subsequent_system_prompt']
                            self.subsequent_task_prompt = prompts['subsequent_task_prompt']
                            logger.info("✅ Emergency prompt reload successful!")
                            print("\n\033[92m✅ Emergency prompt reload successful!\033[0m")
                except Exception as e:
                    logger.critical(f"Emergency reload failed: {e}")
            
            if not hasattr(self, 'first_run_system_prompt') or not self.first_run_system_prompt:
                self.first_run_system_prompt = "You are an AI tasked with cleaning transcriptions from a livestream. Return the response as a JSON object with a single key 'cleaned_text' containing the cleaned transcription."
                
            if not hasattr(self, 'first_run_task_prompt') or not self.first_run_task_prompt:
                self.first_run_task_prompt = "Clean the transcription of Chunk 1 using Chunk 2 as forward context: Chunk 1: {chunk1_text}, Chunk 2: {chunk2_text}"
                
            if not hasattr(self, 'subsequent_system_prompt') or not self.subsequent_system_prompt:
                self.subsequent_system_prompt = "You are an AI tasked with cleaning transcriptions from a livestream. Return the response as a JSON object with a single key 'cleaned_text' containing the cleaned transcription."
                
            if not hasattr(self, 'subsequent_task_prompt') or not self.subsequent_task_prompt:
                self.subsequent_task_prompt = "Clean the current chunk as a continuation: Previous: {prev_cleaned_text}, Current: {raw_text}, Next: {next_raw_text}"
            
        with self.db_lock:
            cursor = self.conn.execute('SELECT text, start_time FROM transcriptions WHERE id = ?', (chunk_id,))
            row = cursor.fetchone()
            if not row:
                logger.error(f"Chunk {chunk_id} not found for cleaning")
                return
            current_raw_text, start_timecode = row

            if self.chunk_count == 1 or chunk_id == 1:  # First chunk case
                next_cursor = self.conn.execute('SELECT text FROM transcriptions WHERE id = ?', (chunk_id + 1,))
                next_row = next_cursor.fetchone()
                if not next_row:
                    logger.debug(f"Next chunk {chunk_id + 1} not available yet to clean chunk {chunk_id}")
                    return
                next_raw_text = next_row[0]
                try:
                    task_prompt = self.first_run_task_prompt.format(
                        chunk1_text=current_raw_text,
                        chunk2_text=next_raw_text
                    )
                except KeyError as e:
                    logger.error(f"Failed to format first_run_task_prompt for chunk {chunk_id}: Missing placeholder '{e}'. Using simple format.")
                    task_prompt = f"Clean this transcription using the next chunk as context.\nChunk 1: {current_raw_text}\nChunk 2: {next_raw_text}"
                
                messages = [
                    {'role': 'system', 'content': self.first_run_system_prompt},
                    {'role': 'user', 'content': task_prompt}
                ]
            else:  # Subsequent chunks
                prev_id = chunk_id - 1
                prev_cleaned_row = self.conn.execute('SELECT cleaned_text FROM transcriptions WHERE id = ?', (prev_id,))
                prev_cleaned = prev_cleaned_row.fetchone()
                if not prev_cleaned or not prev_cleaned[0]:
                    logger.debug(f"Previous chunk {prev_id} not cleaned yet")
                    return
                prev_cleaned_text = prev_cleaned[0]
                
                prev_raw_row = self.conn.execute('SELECT text FROM transcriptions WHERE id = ?', (prev_id,)).fetchone()
                if not prev_raw_row:
                    logger.error(f"Previous raw text for chunk {prev_id} not found")
                    return
                prev_raw_text = prev_raw_row[0]
                
                next_id = chunk_id + 1
                next_raw_row = self.conn.execute('SELECT text FROM transcriptions WHERE id = ?', (next_id,))
                next_raw = next_raw_row.fetchone()
                if not next_raw:
                    logger.debug(f"Next chunk {next_id} not available yet")
                    return
                next_raw_text = next_raw[0]
                try:
                    task_prompt = self.subsequent_task_prompt.format(
                        prev_cleaned_text=prev_cleaned_text,
                        prev_raw_text=prev_raw_text,
                        raw_text=current_raw_text,
                        next_raw_text=next_raw_text
                    )
                except KeyError as e:
                    logger.error(f"Failed to format subsequent_task_prompt for chunk {chunk_id}: Missing placeholder '{e}'. Using simple format.")
                    task_prompt = f"Clean this transcription.\nPrevious cleaned: {prev_cleaned_text}\nPrevious raw: {prev_raw_text}\nCurrent raw: {current_raw_text}\nNext raw: {next_raw_text}"
                    
                messages = [
                    {'role': 'system', 'content': self.subsequent_system_prompt},
                    {'role': 'user', 'content': task_prompt}
                ]

        def ai_clean(messages):
            primary_model = "mistral-small:24b-instruct-2501-q4_K_M"
            fallback_models = [
                "mistral-small", 
                "llama3", 
                "llama2"
            ]
            
            schema_instruction = "\n\nReturn the response as a JSON object with a single key 'cleaned_text' containing the cleaned transcription."
            messages[0]['content'] += schema_instruction
            
            try:
                logger.info(f"Using primary model: {primary_model}")
                response = ollama.chat(
                    messages=messages,
                    model=primary_model,
                    format="json"
                )
                return response
            except Exception as e:
                error_msg = f"⚠️ 🔴 CRITICAL ERROR: Primary model '{primary_model}' failed: {e} 🔴 ⚠️"
                logger.critical(error_msg)
                print(f"\n\033[91m{error_msg}\033[0m")
                print(f"\n\033[91m⚠️ 🔴 IMPORTANT: Run 'ollama pull {primary_model}' to install the recommended model! 🔴 ⚠️\033[0m")
                
                for model in fallback_models:
                    try:
                        logger.info(f"Attempting fallback model: {model}")
                        response = ollama.chat(
                            messages=messages,
                            model=model,
                            format="json"
                        )
                        return response
                    except Exception as fallback_err:
                        logger.warning(f"Fallback model {model} failed: {fallback_err}")
                        continue
                
                logger.critical("❌ All AI models failed! No transcription cleaning possible!")
                error_msg = f"No available AI models found. Please install the recommended model with 'ollama pull {primary_model}'"
                print(f"\n\033[91m⚠️ 🔴 CRITICAL ERROR: {error_msg} 🔴 ⚠️\033[0m")
                raise RuntimeError(error_msg)

        try:
            logger.debug(f"Calling AI to clean chunk {chunk_id}")
            start_time = time.time()
            response = ai_clean(messages)
            end_time = time.time()
            logger.debug(f"AI processing for chunk {chunk_id} took {end_time - start_time:.2f} seconds")
            
            content = response['message']['content']
            try:
                cleaned_data = json.loads(content)
                if 'cleaned_text' in cleaned_data:
                    cleaned_text = cleaned_data['cleaned_text']
                else:
                    logger.warning(f"JSON response missing 'cleaned_text' key: {content}")
                    cleaned_text = content
            except json.JSONDecodeError:
                logger.warning(f"AI response not valid JSON: {content}")
                cleaned_text = content
            
            with self.db_lock:
                self.conn.execute(
                    'UPDATE transcriptions SET cleaned_text = ?, processing_state = ? WHERE id = ?',
                    (cleaned_text, 'cleaned', chunk_id)
                )
                self.conn.commit()
                self.save_transcriptions_to_files("", f"{start_timecode} {cleaned_text}")
            logger.info(f"Successfully cleaned chunk {chunk_id}")
        except Exception as e:
            logger.error(f"Failed to clean chunk {chunk_id}: {e}")
            with self.db_lock:
                self.conn.execute(
                    'UPDATE transcriptions SET cleaned_text = ?, processing_state = ? WHERE id = ?',
                    (current_raw_text, 'raw_only', chunk_id)
                )
                self.conn.commit()
                self.save_transcriptions_to_files("", f"{start_timecode} {current_raw_text}")

    def get_latest_transcriptions(self):
        cursor = self.conn.execute(
            'SELECT start_time, end_time, text, timestamp FROM transcriptions ORDER BY id ASC'
        )
        return cursor.fetchall()

    def get_cleaned_transcriptions(self):
        cursor = self.conn.execute(
            'SELECT start_time, cleaned_text FROM transcriptions WHERE cleaned_text != "" ORDER BY id ASC'
        )
        return cursor.fetchall()

    def save_transcriptions_to_files(self, raw_text: str = "", processed_text: str = "", save_to_files: bool = None):
        if save_to_files is None:
            save_to_files = self.save_texts_to_files
            
        if not save_to_files:
            return
        
        output_dir = "output_text"
        os.makedirs(output_dir, exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        raw_filename = os.path.join(output_dir, f"{date_str}_raw.txt")
        ai_filename = os.path.join(output_dir, f"{date_str}_ai_cleaned.txt")

        if raw_text:
            try:
                with open(raw_filename, 'a', encoding='utf-8') as f:
                    f.write(raw_text + '\n')
                logger.info(f"Updated raw transcription in {raw_filename}")
            except Exception as e:
                logger.error(f"Failed to update raw transcription in {raw_filename}: {e}")

        if processed_text:
            try:
                with open(ai_filename, 'a', encoding='utf-8') as f:
                    f.write(processed_text + '\n')
                logger.info(f"Updated AI-cleaned transcription in {ai_filename}")
            except Exception as e:
                logger.error(f"Failed to update AI-cleaned transcription in {ai_filename}: {e}")

def create_ui(system: TranscriptionSystem):
    def start_transcription(url: str, save_chunks: bool, empty_db: bool, empty_audio_folder: bool, save_texts_to_files: bool, language: str, is_running: bool):
        if not url:
            logger.warning("No URL provided")
            return "Enter a URL", "", gr.update(interactive=True), gr.update(interactive=False), False, "Idle"
        logger.info(
            f"Starting transcription for: {url} (Save chunks: {save_chunks}, Empty DB: {empty_db}, Empty audio folder: {empty_audio_folder}, Save texts to files: {save_texts_to_files}, Language: {language})"
        )
        system.save_texts_to_files = save_texts_to_files
        
        # Apply settings here when starting
        if empty_db:
            system.reset_database()
        if empty_audio_folder:
            try:
                if os.path.exists("audio_chunks"):
                    shutil.rmtree("audio_chunks")
                os.makedirs("audio_chunks", exist_ok=True)
                logger.info("Audio chunks folder cleared")
            except (PermissionError, OSError) as e:
                logger.error(f"Failed to manage audio_chunks directory: {e}")
                raise
        
        thread = threading.Thread(
            target=system.process_stream,
            args=(url, save_chunks, empty_db, empty_audio_folder, language),
            daemon=True
        )
        thread.start()
        return "Transcription started", "", gr.update(interactive=False), gr.update(interactive=True), True, "Running"

    def stop_transcription(is_running: bool):
        if is_running:
            system.stop_process()
            return "Transcription stopped", "", gr.update(interactive=True), gr.update(interactive=False), False, "Idle"
        return "No transcription running", "", gr.update(interactive=True), gr.update(interactive=False), False, "Idle"

    def update_display(is_running: bool, save_texts_to_files: bool):
        if not is_running:
            return "Not started", "Not started"
        transcriptions = system.get_latest_transcriptions()
        raw_text = '\n'.join([
            f"{timestamp} [{start_time}-{end_time}] {text}"
            for start_time, end_time, text, timestamp in transcriptions
        ]) or "No transcriptions yet"
        cleaned_transcriptions = system.get_cleaned_transcriptions()
        processed_text = '\n'.join([
            f"{start_time} {cleaned_text}"
            for start_time, cleaned_text in cleaned_transcriptions
        ]) or "No cleaned transcriptions yet"
        if system.show_ui_update_logs:
            logger.debug("UI updated with latest transcriptions")
        return raw_text, processed_text

    def evaluate_outputs():
        try:
            result = subprocess.run(['python3', 'output_eval.py'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Transcription evaluation triggered successfully")
                return "Evaluation completed successfully - check eval/output_eval_*.txt"
            else:
                logger.error(f"Evaluation failed: {result.stderr}")
                return f"Evaluation failed: {result.stderr}"
        except FileNotFoundError:
            logger.error("output_eval.py not found in current directory")
            return "Error: output_eval.py not found"
        except Exception as e:
            logger.error(f"Unexpected error running evaluation: {e}")
            return f"Error: {str(e)}"

    def get_timer_value(refresh_interval: str) -> Optional[float]:
        if refresh_interval == "none":
            return None
        elif refresh_interval == "10s":
            return 10.0
        elif refresh_interval == "20s":
            return 20.0
        elif refresh_interval == "1m":
            return 60.0
        elif refresh_interval == "5m":
            return 300.0
        elif refresh_interval == "10m":
            return 600.0
        return 60.0

    with gr.Blocks(css="""
        .orange-button {background-color: orange !important; color: white !important;}
        #refresh_btn {background-color: gray !important; color: white !important;}
        #refresh_btn.interactive {background-color: #4CAF50 !important; color: white !important;}
    """) as ui:
        is_running = gr.State(False)
        
        url_input = gr.Textbox(
            label="YouTube URL",
            placeholder="Enter YouTube URL here",
            lines=1,
            show_label=True,
            elem_classes="long-url-bar"
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                save_chunks = gr.Checkbox(label="Save Audio Chunks", value=True)
                empty_db = gr.Checkbox(label="Empty Database", value=True)
                empty_audio_folder = gr.Checkbox(label="Empty Audio Folder", value=True)
                save_texts_to_files = gr.Checkbox(label="Save Texts to Files", value=True)
                language = gr.Dropdown(
                    label="Transcription Language",
                    choices=["Korean", "English", "Chinese", "Vietnamese", "Spanish"],
                    value="Korean",
                    info="Select the language for transcription."
                )
                refresh_interval = gr.Dropdown(
                    label="Refresh Interval",
                    choices=["none", "10s", "20s", "1m", "5m", "10m"],
                    value="1m",
                    info="Select how often the text boxes refresh automatically."
                )
            
            with gr.Column(scale=1):
                start_btn = gr.Button("Start Transcription", elem_classes="orange-button")
                refresh_btn = gr.Button("Refresh Transcriptions", elem_id="refresh_btn", interactive=False)
                stop_btn = gr.Button("Stop Transcription")
                eval_btn = gr.Button("Evaluate Outputs")

        raw_output = gr.Textbox(label="Raw Transcription", lines=20, max_lines=10000)
        processed_output = gr.Textbox(label="Processed Transcription (Ollama)", lines=20, max_lines=10000)
        eval_status = gr.Textbox(label="Evaluation Status", value="Idle", interactive=False)

        start_btn.click(
            fn=start_transcription,
            inputs=[url_input, save_chunks, empty_db, empty_audio_folder, save_texts_to_files, language, is_running],
            outputs=[raw_output, processed_output, refresh_btn, stop_btn, is_running, eval_status]
        )
        refresh_btn.click(
            fn=update_display,
            inputs=[is_running, save_texts_to_files],
            outputs=[raw_output, processed_output]
        )
        stop_btn.click(
            fn=stop_transcription,
            inputs=[is_running],
            outputs=[raw_output, processed_output, refresh_btn, stop_btn, is_running, eval_status]
        )
        eval_btn.click(
            fn=evaluate_outputs,
            inputs=[],
            outputs=[eval_status]
        )
        timer = gr.Timer(value=get_timer_value(refresh_interval.value))
        timer.tick(
            fn=update_display,
            inputs=[is_running, save_texts_to_files],
            outputs=[raw_output, processed_output]
        )
        
        def update_timer_interval(interval_value):
            return gr.Timer(value=get_timer_value(interval_value))
        
        refresh_interval.change(
            fn=update_timer_interval,
            inputs=[refresh_interval],
            outputs=[timer]
        )

    return ui

def main():
    model_path = "whisper.cpp/models/ggml-large-v3-turbo.bin"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        print(f"ERROR: Model file not found: {model_path}")
        print("Please download the model using whisper.cpp/models/download-ggml-model.sh")
        return
        
    system = TranscriptionSystem(model_path, show_ui_update_logs=False)
    ui = create_ui(system)
    
    ui.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()