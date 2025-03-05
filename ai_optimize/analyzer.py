import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal, Union, Tuple
from pydantic import BaseModel, Field
import time
import re
from ollama import AsyncClient
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ai_analysis.log")
    ]
)
logger = logging.getLogger("ai_optimize.analyzer")

# Constants for real vs example runs
AUDIO_AI_OPTIMIZED_DIR = "audio_ai_optimized"
AI_ANALYSIS_DIR = os.path.join(AUDIO_AI_OPTIMIZED_DIR, "analysis")

# Helper function for robust JSON parsing
def robust_json_parse(text: str) -> Union[Dict[str, Any], None]:
    """
    Robustly parse JSON from text, handling common formatting issues
    from language model outputs.
    
    Args:
        text: String containing JSON (possibly with formatting issues)
        
    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    if not text:
        logger.error("Empty text provided to robust_json_parse")
        return None
        
    logger.debug(f"Attempting to parse JSON from text: {text[:200]}...")
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parsing failed: {str(e)}")
        
        # More aggressive markdown code block removal (handles various formats)
        # Remove code block markers if present (including ```json, ```javascript, etc.)
        cleaned = re.sub(r'^```(?:json|javascript|js)?\s*', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        # Handle cases where there might be text before or after the JSON block
        json_block_match = re.search(r'```(?:json|javascript|js)?\s*([\s\S]*?)\s*```', text, flags=re.IGNORECASE)
        if json_block_match:
            logger.debug("Found JSON block within markdown code blocks")
            cleaned = json_block_match.group(1)
        
        logger.debug(f"After markdown cleanup: {cleaned[:200]}...")
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing after markdown cleanup failed: {str(e)}")
            
            # Fix trailing commas in objects and arrays
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            # Try again
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing after fixing trailing commas failed: {str(e)}")
                
                # Convert Python-style True/False/None to JSON
                cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
                cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
                cleaned = re.sub(r'\bNone\b', 'null', cleaned)
                
                # Replace single quotes with double quotes (if not within double quotes)
                single_to_double = ""
                in_double_quotes = False
                for char in cleaned:
                    if char == '"':
                        in_double_quotes = not in_double_quotes
                    if char == "'" and not in_double_quotes:
                        single_to_double += '"'
                    else:
                        single_to_double += char
                
                logger.debug(f"After quote and boolean conversion: {single_to_double[:200]}...")
                
                try:
                    return json.loads(single_to_double)
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parsing after quote conversion failed: {str(e)}")
                    
                    # Try to extract JSON object using regex
                    json_match = re.search(r'(\{[\s\S]*\})', single_to_double)
                    if json_match:
                        try:
                            potential_json = json_match.group(1)
                            logger.debug(f"Extracted potential JSON object: {potential_json[:200]}...")
                            return json.loads(potential_json)
                        except json.JSONDecodeError as e:
                            logger.debug(f"JSON parsing of extracted object failed: {str(e)}")
                            
                            # Fix unquoted keys
                            quoted_keys = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_match.group(1))
                            logger.debug(f"After quoting keys: {quoted_keys[:200]}...")
                            
                            try:
                                return json.loads(quoted_keys)
                            except json.JSONDecodeError as e:
                                logger.debug(f"JSON parsing after quoting keys failed: {str(e)}")
                                logger.error("All JSON parsing attempts failed. Returning None.")
                                return None
                    
                    # If we couldn't extract a JSON object
                    logger.error("Could not extract JSON object. Returning None.")
                    return None

# Pydantic models for structured output
class AnalysisStep(BaseModel):
    """Represents a single step in the analysis process"""
    step_number: int
    description: str
    observation: str

class GradingResult(BaseModel):
    """Represents the grading result for transcription analysis"""
    reasoning_steps: List[AnalysisStep] = Field(..., min_items=1, description="Step by step reasoning process with detailed explanations")
    verbatim_match_score: int = Field(..., ge=0, le=10, description="How well the transcription content matches verbatim at the chunk boundaries")
    sentence_preservation_score: int = Field(..., ge=0, le=10, description="How well sentences are preserved across chunk boundaries")
    content_duplication_score: int = Field(..., ge=0, le=10, description="How much content is unnecessarily duplicated at chunk boundaries")
    content_loss_score: int = Field(..., ge=0, le=10, description="How much content is missing at chunk boundaries")
    join_transition_score: int = Field(..., ge=0, le=10, description="How smooth the transition is between chunks")
    contextual_flow_score: int = Field(..., ge=0, le=10, description="How well the meaning and context flow across chunk boundaries")
    
    @property
    def average_score(self) -> float:
        """Calculate the average score across all grading criteria"""
        total = (
            self.verbatim_match_score + 
            self.sentence_preservation_score + 
            self.content_duplication_score + 
            self.content_loss_score + 
            self.join_transition_score + 
            self.contextual_flow_score
        )
        return round(total / 6, 2)

class TranscriptionAnalyzer:
    """Analyzer for evaluating transcription quality"""
    
    def __init__(self, model_name="mistral-small:24b-instruct-2501-q4_K_M", language="en"):
        """
        Initialize the analyzer with the specified AI model and language.
        
        Args:
            model_name: Name of the model to use for analysis
            language: Language code (e.g., "en", "kr") for selecting prompt files
        """
        self.model_name = model_name
        self.language = language
        self.prompts = self._load_prompts()
        logger.info(f"Initialized TranscriptionAnalyzer with model: {model_name}, language: {language}")
        
    @staticmethod
    def get_analysis_dir(dir_path=None):
        """
        Get the correct analysis directory, handling both example and real runs.
        
        Args:
            dir_path: Optional directory path to check
            
        Returns:
            The appropriate analysis directory path
        """
        if dir_path is None or not "example" in dir_path.lower():
            return AI_ANALYSIS_DIR
        return dir_path
    
    def _load_prompts(self) -> Dict[str, str]:
        """
        Load the prompts from the JSON file based on the language.
        Falls back to default English prompts if language-specific prompts not found.
        """
        # Determine the prompt file path based on language
        prompt_file = f"ai_optimize/ai_optimizer_prompts_{self.language}.json"
        default_prompt_file = "ai_optimize/ai_optimizer_prompts.json"
        
        # Add more detailed logging
        logger.warning(f"*** PROMPT FILE LOADING DEBUG ***")
        logger.warning(f"* Language code: {self.language}")
        logger.warning(f"* Trying to load prompt file: {prompt_file}")
        logger.warning(f"* Prompt file exists: {os.path.exists(prompt_file)}")
        logger.warning(f"* Default prompt file exists: {os.path.exists(default_prompt_file)}")
        
        try:
            # First try loading language-specific prompts
            if os.path.exists(prompt_file):
                with open(prompt_file, "r") as f:
                    prompts = json.load(f)
                logger.info(f"Successfully loaded analysis prompts for language: {self.language}")
            else:
                # If language-specific prompts don't exist and it's not English, error out
                if self.language != "en":
                    error_msg = f"Prompt file for language '{self.language}' not found at {prompt_file}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                # For English, use the default prompts
                with open(default_prompt_file, "r") as f:
                    prompts = json.load(f)
                logger.info("Successfully loaded default (English) analysis prompts")
            
            # Add detailed debugging about loaded prompts
            logger.debug(f"Prompts loaded: {list(prompts.keys())}")
            for key, value in prompts.items():
                logger.debug(f"  - {key}: length {len(value)} chars, starts with: {value[:50]}...")
                
                # Check for placeholder formatting
                placeholders = re.findall(r'\{\{([^}]+)\}\}', value)
                if placeholders:
                    logger.debug(f"  - Contains placeholders: {placeholders}")
                else:
                    logger.debug(f"  - No placeholders found!")
                
            return prompts
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
    
    async def analyze_chunk(self, 
                           current_chunk: str, 
                           previous_chunk: Optional[str] = None,
                           next_chunk: Optional[str] = None,
                           prompt_used: Optional[str] = None,
                           overlap_seconds: int = 5) -> Tuple[GradingResult, Dict[str, Any]]:
        """Analyze a transcription chunk using AI
        
        Args:
            current_chunk: The current transcription chunk to analyze
            previous_chunk: The previous transcription chunk (if available)
            next_chunk: The next transcription chunk (if available)
            prompt_used: The prompt used for transcription (if available)
            overlap_seconds: Duration of overlap padding in seconds
            
        Returns:
            Tuple containing:
            - GradingResult object containing the analysis and scores
            - Dict with the raw Ollama response
        """
        # Verify input parameters
        if current_chunk is None or not current_chunk.strip():
            error_msg = "Current chunk is empty or None"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Default values for optional parameters
        previous_chunk = previous_chunk or "No previous chunk available"
        next_chunk = next_chunk or "No next chunk available"
        prompt_used = prompt_used or "No prompt information available"
        
        # Create a default chunk info if not already set
        if not hasattr(self, 'current_chunk_info') or not self.current_chunk_info:
            # Create a simple identifier for the chunk
            chunk_id = hashlib.md5(current_chunk[:100].encode()).hexdigest()[:8]
            self.current_chunk_info = {
                "chunk_number": 0,
                "filename": f"direct_chunk_{chunk_id}",
                "audio_file": "unknown"
            }
            
            # If there's no params dir set, create a default one
            if not hasattr(self, 'current_params_dir') or not self.current_params_dir:
                analysis_dir = self.get_analysis_dir()
                self.current_params_dir = os.path.join(analysis_dir, "direct_analysis")
                os.makedirs(self.current_params_dir, exist_ok=True)
        
        # Print the first 100 chars of each chunk for debugging
        logger.debug(f"CHUNK VALUES:")
        logger.debug(f"  - current_chunk: {current_chunk[:100]}... (length: {len(current_chunk)})")
        logger.debug(f"  - previous_chunk: {previous_chunk[:100]}... (length: {len(previous_chunk)})")
        logger.debug(f"  - next_chunk: {next_chunk[:100]}... (length: {len(next_chunk)})")
        logger.debug(f"  - prompt_used: {prompt_used[:100]}... (length: {len(prompt_used)})")
        logger.debug(f"  - overlap_seconds: {overlap_seconds}")
        
        # Determine which prompt to use based on available context and overlap seconds
        if previous_chunk == "No previous chunk available" and next_chunk != "No next chunk available":
            # First chunk
            if overlap_seconds == 0:
                template_key = "grading_user_prompt_first_chunk_zero_overlap"
            else:
                template_key = "grading_user_prompt_first_chunk"
            context_type = "first chunk"
        elif previous_chunk != "No previous chunk available" and next_chunk == "No next chunk available":
            # Last chunk
            if overlap_seconds == 0:
                template_key = "grading_user_prompt_last_chunk_zero_overlap"
            else:
                template_key = "grading_user_prompt_last_chunk"
            context_type = "last chunk"
        else:
            # Middle chunk
            if overlap_seconds == 0:
                template_key = "grading_user_prompt_standard_zero_overlap"
            else:
                template_key = "grading_user_prompt_standard"
            context_type = "middle chunk"
        
        # Add more detailed logging about language and template selection
        logger.warning(f"*** PROMPT TEMPLATE SELECTION DEBUG ***")
        logger.warning(f"* Language code: {self.language}")
        logger.warning(f"* Context type: {context_type}")
        logger.warning(f"* Selected template key: {template_key}")
        logger.warning(f"* Available templates: {list(self.prompts.keys())}")
        
        # Validate that the template exists in our prompts
        if template_key not in self.prompts:
            logger.error(f"Template key '{template_key}' not found in loaded prompts! Available keys: {list(self.prompts.keys())}")
            raise ValueError(f"Missing template: {template_key}")
        
        prompt_template = self.prompts[template_key]
        logger.debug(f"Using {context_type} template ({template_key}) with length: {len(prompt_template)}")
        
        # Log the template before substitution
        logger.debug(f"Template before substitution (first 200 chars): {prompt_template[:200]}...")
        
        # Direct substitution: Instead of using string.replace() which can be problematic with nested placeholders,
        # we'll format the entire template at once with the specific values
        try:
            # Create a dictionary mapping placeholders to values
            replacements = {
                "{{current_chunk}}": current_chunk,
                "{{previous_chunk}}": previous_chunk,
                "{{next_chunk}}": next_chunk,
                "{{prompt_used}}": prompt_used,
                "{{overlap_seconds}}": str(overlap_seconds)
            }
            
            # Check which placeholders are actually in the template
            for placeholder in replacements.keys():
                if placeholder in prompt_template:
                    logger.debug(f"  ✓ Template contains {placeholder}")
                else:
                    logger.warning(f"  ✗ Template DOES NOT contain {placeholder}! Substitution will have no effect.")
            
            # Perform replacements
            prompt = prompt_template
            for placeholder, replacement in replacements.items():
                if placeholder in prompt:
                    logger.debug(f"Replacing {placeholder} with text of length {len(replacement)}")
                    prompt = prompt.replace(placeholder, replacement)
                
            logger.debug(f"After direct substitution, prompt begins with: {prompt[:200]}...")
        except Exception as e:
            # Log error if substitution fails
            logger.error(f"Error with substitution: {str(e)}.")
            raise

        # VERIFICATION: Check if any placeholders remain unsubstituted
        placeholder_pattern = r'\{\{([^}]+)\}\}'
        remaining_placeholders = re.findall(placeholder_pattern, prompt)
        
        if remaining_placeholders:
            # Log detailed information about unsubstituted placeholders
            logger.error(f"PROMPT ERROR: Found {len(remaining_placeholders)} unsubstituted placeholders in prompt")
            for placeholder in remaining_placeholders:
                logger.error(f"  - Unsubstituted placeholder: {{{{{placeholder}}}}}")
            
            # Log the state of the variables
            logger.error("Variable values:")
            logger.error(f"  - current_chunk: {current_chunk[:50]}... (length: {len(current_chunk)})")
            logger.error(f"  - previous_chunk: {previous_chunk[:50]}... (length: {len(previous_chunk)})")
            logger.error(f"  - next_chunk: {next_chunk[:50]}... (length: {len(next_chunk)})")
            logger.error(f"  - prompt_used: {prompt_used[:50]}... (length: {len(prompt_used)})")
            
            # Check if the template actually contained the placeholders
            logger.error("Original template analysis:")
            for placeholder in remaining_placeholders:
                placeholder_full = f"{{{{{placeholder}}}}}"
                if placeholder_full in prompt_template:
                    logger.error(f"  - Template contained {placeholder_full} but replacement failed")
                else:
                    logger.error(f"  - Template did NOT contain {placeholder_full} - possible dynamic injection?")
            
            # Try to fix the issue by substituting default values for any remaining placeholders
            for placeholder in remaining_placeholders:
                default_value = f"[Missing {placeholder}]"
                placeholder_pattern_str = r'\{\{' + re.escape(placeholder) + r'\}\}'
                prompt = re.sub(placeholder_pattern_str, default_value, prompt)
                logger.warning(f"  - Substituted default value for {{{{{placeholder}}}}}: '{default_value}'")
        
        system_prompt = self.prompts["grading_system_prompt"]
        
        logger.info(f"Analyzing chunk with model: {self.model_name}")
        logger.debug(f"System prompt length: {len(system_prompt)} characters")
        logger.debug(f"User prompt length: {len(prompt)} characters")
        logger.debug(f"System prompt (first 100 chars): {system_prompt[:100]}...")
        logger.debug(f"User prompt (first 100 chars): {prompt[:100]}...")
        
        # Check if key sections are present
        logger.debug(f"Prompt contains 'Current Chunk': {'Current Chunk' in prompt}")
        logger.debug(f"Prompt contains 'Previous Chunk': {'Previous Chunk' in prompt}")
        logger.debug(f"Prompt contains 'Next Chunk': {'Next Chunk' in prompt}")
        
        # Add detailed content verification
        def extract_chunk_content(section_name, prompt_text):
            pattern = f"{section_name}:\\n(.*?)\\n\\n"
            match = re.search(pattern, prompt_text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                return content, len(content)
            return "Section not found", 0
        
        current_content, current_len = extract_chunk_content("Current Chunk", prompt)
        prev_content, prev_len = extract_chunk_content("Previous Chunk", prompt)
        next_content, next_len = extract_chunk_content("Next Chunk", prompt)
        
        logger.debug(f"  - Current Chunk content length: {current_len} chars")
        logger.debug(f"  - Previous Chunk content length: {prev_len} chars")
        logger.debug(f"  - Next Chunk content length: {next_len} chars")
        
        # Check for potentially problematic content
        if current_len < 5 and "Current Chunk" in prompt:
            logger.warning(f"Current Chunk section appears to be empty or very short: '{current_content}'")
        if prev_len < 5 and "Previous Chunk" in prompt:
            logger.warning(f"Previous Chunk section appears to be empty or very short: '{prev_content}'")
        if next_len < 5 and "Next Chunk" in prompt:
            logger.warning(f"Next Chunk section appears to be empty or very short: '{next_content}'")
            
        # Set up retry parameters
        max_retries = 3
        retry_count = 0
        backoff_factor = 2  # Exponential backoff
        
        while retry_count < max_retries:
            try:
                # Create a client for this request
                client = AsyncClient()
                
                # Log the request details
                logger.info(f"Sending request to Ollama API with model {self.model_name} (attempt {retry_count + 1}/{max_retries})")
                
                # Get the JSON schema from the Pydantic model
                schema = GradingResult.model_json_schema()
                
                # Prepare request parameters
                request_params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "format": schema,  # Pass the schema directly here
                    "options": {
                        "temperature": 0.5,  # Use temperature 0 for more deterministic JSON outputs
                        "num_predict": 2048  # Ensure enough tokens for complete response
                    }
                }
                
                # VERIFICATION: Check that the prompt doesn't contain placeholder patterns before sending
                final_prompt_check = re.findall(r'\{\{([^}]+)\}\}', prompt)
                if final_prompt_check:
                    logger.error(f"FINAL VERIFICATION FAILED: Prompt still contains placeholders after attempted fixes: {final_prompt_check}")
                
                # Log the FULL request for debugging
                logger.info("================ FULL OLLAMA REQUEST ================")
                logger.info(f"Model: {self.model_name}")
                logger.info(f"System prompt: {system_prompt}")
                logger.info(f"User prompt: {prompt}")
                logger.info(f"Schema: {json.dumps(schema, default=str)[:500]}...")
                logger.info(f"Options: {json.dumps(request_params.get('options', {}))}")
                logger.info("====================================================")
                
                # Save the prompt to a text file in the working folder for this run
                if hasattr(self, 'current_chunk_info') and self.current_chunk_info and hasattr(self, 'current_params_dir') and self.current_params_dir:
                    try:
                        # Get relevant information
                        chunk_number = self.current_chunk_info.get('chunk_number', 'unknown')
                        # Create a clean filename from the chunk name and combination
                        chunk_name = self.current_chunk_info.get('filename', f"chunk_{chunk_number}")
                        
                        # Extract combination name from params_dir if available
                        combination = os.path.basename(self.current_params_dir)
                        if combination == "raw_responses" or combination == "direct_analysis":
                            combination = "direct"
                        
                        # Extract run timestamp from directory path - typically in format "run_YYYYMMDD_HHMMSS"
                        run_timestamp = ""
                        parent_dirs = self.current_params_dir.split(os.path.sep)
                        for dir_name in parent_dirs:
                            if dir_name.startswith("run_") and len(dir_name) > 15:  # run_ plus YYYYMMDD_HHMMSS
                                run_timestamp = dir_name[4:]  # Remove "run_" prefix
                                break
                        
                        # If no run timestamp found, use current time as fallback
                        if not run_timestamp:
                            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Create a unique filename including real chunk number, combination, and language
                        language_suffix = f"_{self.language}" if hasattr(self, 'language') and self.language else ""
                        
                        # Use chunk_name which might contain the real chunk number from the filename
                        # Extract chunk number from filename if present (e.g., "chunk_5" -> "5")
                        chunk_id = "0"
                        if isinstance(chunk_name, str):
                            chunk_match = re.search(r'chunk[_]?(\d+)', chunk_name)
                            if chunk_match:
                                chunk_id = chunk_match.group(1)
                            # Also check if filename contains a number directly
                            elif re.search(r'\d+', chunk_name):
                                chunk_id = re.search(r'(\d+)', chunk_name).group(1)
                        
                        prompt_filename = f"{combination}_chunk{chunk_id}{language_suffix}_{run_timestamp}_prompt.txt"
                        prompt_path = os.path.join(self.current_params_dir, prompt_filename)
                        
                        # Write both system and user prompts to the file
                        with open(prompt_path, 'w', encoding='utf-8') as f:
                            f.write("===== SYSTEM PROMPT =====\n\n")
                            f.write(system_prompt)
                            f.write("\n\n===== USER PROMPT =====\n\n")
                            f.write(prompt)
                            
                        logger.info(f"Saved prompt to {prompt_path}")
                    except Exception as e:
                        logger.error(f"Error saving prompt to file: {str(e)}")
                
                # Make the API call with a timeout
                logger.debug(f"Request parameters: {json.dumps(request_params, default=str)[:500]}...")

                response = await asyncio.wait_for(
                    client.chat(**request_params),
                    timeout=60  # 60 second timeout
                )
                
                # Create a raw response object with request details
                raw_response = {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_name,
                    "request": {
                        "messages": [
                            {"role": msg["role"], "content_preview": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]} 
                            for msg in request_params.get("messages", [])
                        ],
                        "options": request_params.get("options", {}),
                        "format": request_params.get("format", None)
                    },
                    "response": response
                }
                
                # Log the FULL raw response
                logger.info("================ FULL OLLAMA RESPONSE ================")
                logger.info(f"Response type: {type(response)}")
                logger.info(f"Response raw: {response}")
                if isinstance(response, dict):
                    logger.info(f"Response keys: {list(response.keys())}")
                    if 'message' in response:
                        logger.info(f"Message type: {type(response['message'])}")
                        logger.info(f"Message content: {response.get('message', {}).get('content', 'N/A')}")
                else:
                    logger.info(f"Response string repr: {str(response)}")
                logger.info("=====================================================")
                
                # Validate the response contains expected attributes
                if not response:
                    raise ValueError("Received empty response from Ollama API")

                # Extract content from response - standardized handling for Ollama responses
                content = None
                
                # Handle the updated Ollama API response format 
                if isinstance(response, dict):
                    # Most common Ollama API response format (message field containing content)
                    if 'message' in response and isinstance(response['message'], dict) and 'content' in response['message']:
                        content = response['message']['content']
                    # Alternative format (direct content field)
                    elif 'content' in response:
                        content = response['content']
                    # JSON format response (may contain direct JSON)
                    elif 'response' in response:
                        content = response['response']
                    
                    # Log which field we're using
                    if content:
                        logger.info(f"Using content from field: {'message.content' if 'message' in response else ('content' if 'content' in response else 'response')}")
                
                if content is None:
                    logger.error(f"Could not extract content from response: {response}")
                    raise ValueError(f"Failed to extract content from Ollama response. Response structure: {json.dumps(response, default=str) if isinstance(response, dict) else str(response)}")
                
                # Log the raw content for debugging
                logger.info("================ EXTRACTED CONTENT ================")
                logger.info(f"Content (first 500 chars): {content[:500]}")
                if len(content) > 500:
                    logger.info(f"... (truncated, total length: {len(content)} chars)")
                logger.info("==================================================")
                
                # Save the original content before any processing
                raw_response["original_content"] = content
                
                if not content.strip():
                    raise ValueError("Message content is empty")
                
                try:
                    # First try using our robust JSON parser to handle various formatting issues
                    logger.info("Attempting to parse content with robust_json_parse")
                    json_data = robust_json_parse(content)
                    
                    if not json_data:
                        logger.warning("robust_json_parse failed, trying to extract JSON from potential text")
                        # Try to find JSON object in the content if it's mixed with non-JSON text
                        json_matches = re.finditer(r'(\{[^{]*"verbatim_match_score"[^}]*\})', content)
                        for match in json_matches:
                            potential_json = match.group(1)
                            logger.info(f"Found potential JSON object: {potential_json[:200]}...")
                            try:
                                json_data = json.loads(potential_json)
                                if "verbatim_match_score" in json_data:
                                    logger.info("Successfully extracted JSON from content")
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    # If we still don't have valid JSON data
                    if not json_data:
                        raise ValueError(f"Failed to parse JSON from response content: {content[:200]}...")
                    
                    # Ensure all required fields are present
                    required_fields = ["reasoning_steps", "verbatim_match_score", "sentence_preservation_score", "content_duplication_score",
                                      "content_loss_score", "join_transition_score", "contextual_flow_score"]
                    
                    missing_fields = [field for field in required_fields if field not in json_data]
                    
                    # Special handling for missing reasoning_steps - add a default one
                    if "reasoning_steps" in missing_fields:
                        logger.warning("Response missing reasoning_steps. Adding default reasoning step.")
                        json_data["reasoning_steps"] = [
                            {
                                "step_number": 1,
                                "description": "Overall analysis",
                                "observation": "Model did not provide detailed reasoning steps. Scores are still valid."
                            }
                        ]
                        missing_fields.remove("reasoning_steps")
                    
                    # Check if there are still missing fields
                    if missing_fields:
                        raise ValueError(f"JSON response missing required fields: {missing_fields}")
                    
                    # Create GradingResult object
                    result = GradingResult(**json_data)
                    logger.info(f"Successfully analyzed chunk with model {self.model_name}. Average score: {result.average_score}")
                    
                    return result, raw_response
                
                except Exception as parse_error:
                    logger.error(f"Error parsing Ollama response: {str(parse_error)}")
                    logger.debug(f"Raw response content: {content}")
                    raise ValueError(f"Failed to parse Ollama response: {str(parse_error)}")
                
            except asyncio.TimeoutError:
                retry_count += 1
                wait_time = backoff_factor ** retry_count
                logger.warning(f"Ollama API request timed out. Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
            
            except ValueError as ve:
                # For value errors (parsing issues), we retry
                retry_count += 1
                wait_time = backoff_factor ** retry_count
                logger.warning(f"Value error: {str(ve)}. Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                # Check if it's a connection error
                error_msg = str(e).lower()
                if "connection" in error_msg or "network" in error_msg or "refused" in error_msg:
                    retry_count += 1
                    wait_time = backoff_factor ** retry_count
                    logger.warning(f"Ollama API connection error: {str(e)}. Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    # For non-connection errors, check specific cases
                    if "connection refused" in error_msg or "failed to connect" in error_msg:
                        error_details = "It appears Ollama is not running. Please start Ollama and try again."
                    elif "no such model" in error_msg or "model not found" in error_msg:
                        error_details = f"The model '{self.model_name}' was not found. Please make sure it's installed with 'ollama pull {self.model_name}'."
                    else:
                        error_details = "An unexpected error occurred when calling Ollama."
                    
                    logger.error(f"Ollama API error: {error_details}\nOriginal error: {str(e)}")
                    raise Exception(f"{error_details}\nOriginal error: {str(e)}")
        
        # If we've exhausted all retries
        logger.error(f"Failed to analyze chunk after {max_retries} attempts")
        raise RuntimeError(f"Failed to analyze chunk after {max_retries} attempts. Last error was related to: {error_msg if 'error_msg' in locals() else 'unknown issue'}")
    
    async def analyze_batch(self, 
                         transcriptions: List[Dict[str, Any]], 
                         output_dir: str,
                         params_name: str = "",
                         overlap_seconds: int = 5) -> Dict[str, Any]:
        """
        Analyze a batch of transcription chunks and return a summary of results.
        
        Args:
            transcriptions: List of transcription chunks with metadata
            output_dir: Directory to save analysis results
            params_name: Name of the parameter group for organizing results
            overlap_seconds: Number of seconds that overlap between consecutive chunks

        Returns:
            Dictionary with summary statistics and detailed results
        """
        # Determine if this is an example run or a real run
        is_example_run = "example" in output_dir.lower()
        
        # For real runs, use the AI_ANALYSIS_DIR unless explicitly overridden
        if not is_example_run:
            real_output_dir = AI_ANALYSIS_DIR
            logger.info(f"Using real output directory: {real_output_dir} instead of {output_dir}")
            output_dir = real_output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        analysis_dir = os.path.join(output_dir, "analysis")
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
            
        results = []
        raw_responses = []
        
        # Create analysis directory if it doesn't exist
        analysis_dir = os.path.join(output_dir, "analysis")
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
            
        # Create raw_responses directory for storing individual responses
        raw_responses_dir = os.path.join(analysis_dir, "raw_responses")
        if not os.path.exists(raw_responses_dir):
            os.makedirs(raw_responses_dir)
        
        # If params_name is provided, create a subdirectory for this parameter set
        if params_name:
            params_dir = os.path.join(raw_responses_dir, params_name)
            if not os.path.exists(params_dir):
                os.makedirs(params_dir)
        else:
            params_dir = raw_responses_dir
        
        logger.info(f"Starting batch analysis of {len(transcriptions)} chunks")
        
        # Sort transcriptions by chunk number to ensure correct order
        sorted_transcriptions = sorted(transcriptions, key=lambda x: x.get("chunk_number", 0))
        
        # Create a detailed log file for this analysis run
        analysis_log_path = os.path.join(analysis_dir, "detailed_analysis.log")
        with open(analysis_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total chunks to analyze: {len(sorted_transcriptions)}\n")
            f.write(f"Parameter set: {params_name}\n")
            f.write("=" * 50 + "\n\n")
        
        # VALIDATION: Check all transcriptions for valid content before starting
        invalid_transcriptions = []
        for i, chunk_info in enumerate(sorted_transcriptions):
            chunk_number = chunk_info.get("chunk_number", i)
            transcription = chunk_info.get("transcription", "")
            filename = chunk_info.get("filename", f"chunk_{chunk_number}")
            
            if not transcription or not transcription.strip():
                invalid_transcriptions.append({
                    "index": i,
                    "chunk_number": chunk_number,
                    "filename": filename,
                    "issue": "Empty transcription"
                })
            elif len(transcription.strip()) < 10:
                invalid_transcriptions.append({
                    "index": i,
                    "chunk_number": chunk_number,
                    "filename": filename,
                    "issue": f"Very short transcription ({len(transcription.strip())} chars): '{transcription.strip()}'"
                })
                
        if invalid_transcriptions:
            logger.warning(f"Found {len(invalid_transcriptions)} invalid transcriptions before analysis")
            with open(analysis_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\nWARNING: Found {len(invalid_transcriptions)} invalid transcriptions:\n")
                for invalid in invalid_transcriptions:
                    logger.warning(f"  - Chunk {invalid['chunk_number']} ({invalid['filename']}): {invalid['issue']}")
                    f.write(f"  - Chunk {invalid['chunk_number']} ({invalid['filename']}): {invalid['issue']}\n")
        
        # Create a list to store all raw responses for this parameter set
        all_raw_responses = []
        all_grading_results = []
        
        # Tracking metrics
        total_files = len(sorted_transcriptions)
        successful_files = 0
        failed_files = 0
        retried_files = 0
        skipped_files = 0
        
        # Store the current params directory for prompt saving
        self.current_params_dir = params_dir
        
        for i, transcription_data in enumerate(sorted_transcriptions):
            # Get transcription data
            transcription = transcription_data.get("transcription", "").strip()
            chunk_number = transcription_data.get("chunk_number", i)
            filename = transcription_data.get("filename", f"chunk_{chunk_number}")
            audio_file = transcription_data.get("audio_file", "unknown_file")
            prompt_used = transcription_data.get("prompt", "")
            
            # Store the current chunk info for prompt saving
            self.current_chunk_info = {
                "chunk_number": chunk_number,
                "filename": filename,
                "audio_file": audio_file
            }
            
            # Use overlap_seconds from the transcription data if available, otherwise use the provided value
            chunk_overlap_seconds = transcription_data.get("overlap_seconds", overlap_seconds)
            
            # Log the start of processing for this chunk
            logger.info(f"Processing chunk {chunk_number} ({i+1}/{len(sorted_transcriptions)}): {filename}")
            with open(analysis_log_path, 'a', encoding='utf-8') as f:
                f.write(f"Processing chunk {chunk_number} ({i+1}/{len(sorted_transcriptions)}): {filename}\n")
                f.write(f"Audio file: {audio_file}\n")
                f.write(f"Transcription length: {len(transcription)} characters\n")
                if prompt_used:
                    f.write(f"Prompt used: {prompt_used[:100]}{'...' if len(prompt_used) > 100 else ''}\n")
            
            # Get previous and next chunks if available
            previous_chunk = None
            next_chunk = None
            
            if i > 0:
                previous_chunk = sorted_transcriptions[i-1].get("transcription", "")
            
            if i < len(sorted_transcriptions) - 1:
                next_chunk = sorted_transcriptions[i+1].get("transcription", "")
            
            # Track retries for this file
            retries_for_file = 0
            
            try:
                # Log analysis attempt
                with open(analysis_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"Starting analysis with model: {self.model_name}\n")
                    f.write(f"Context: {'First chunk' if previous_chunk is None and next_chunk is not None else 'Last chunk' if previous_chunk is not None and next_chunk is None else 'Middle chunk'}\n")
                
                # Analyze the chunk
                analysis_start_time = time.time()
                
                max_retries = 3
                retry_count = 0
                last_error = None
                
                while retry_count < max_retries:
                    try:
                        analysis_result, raw_response = await self.analyze_chunk(
                            current_chunk=transcription,
                            previous_chunk=previous_chunk,
                            next_chunk=next_chunk,
                            prompt_used=prompt_used,
                            overlap_seconds=chunk_overlap_seconds
                        )
                        
                        # If we had to retry, increment the counter
                        if retry_count > 0:
                            retried_files += 1
                            retries_for_file = retry_count
                        
                        # Success - break out of retry loop
                        break
                    except Exception as e:
                        retry_count += 1
                        last_error = e
                        
                        # Only wait and retry if we haven't exceeded max retries
                        if retry_count < max_retries:
                            wait_time = 2 ** retry_count  # Exponential backoff
                            logger.warning(f"Retry {retry_count}/{max_retries} for chunk {chunk_number} after error: {str(e)}. Waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            # Max retries exceeded, re-raise the exception to be caught below
                            raise
                
                analysis_duration = time.time() - analysis_start_time
                
                # Add to our list of raw responses
                audio_file_basename = os.path.basename(audio_file) if isinstance(audio_file, str) else str(audio_file)
                
                # Create a comprehensive response record
                raw_response_with_metadata = {
                    "chunk_number": chunk_number,
                    "filename": filename,
                    "audio_file": audio_file_basename,
                    "timestamp": datetime.now().isoformat(),
                    "analysis_duration_seconds": round(analysis_duration, 2),
                    "retries": retries_for_file,
                    "raw_response": raw_response,
                    "original_model_content": raw_response.get("original_content", "Not available")
                }
                all_raw_responses.append(raw_response_with_metadata)
                
                # Add the result to our list of grading results
                # Check if analysis_result is a tuple instead of a GradingResult
                if isinstance(analysis_result, tuple):
                    logger.warning(f"Received tuple instead of GradingResult object. Using first element of tuple.")
                    # If it's a tuple, assume the first element is the GradingResult
                    grading_obj = analysis_result[0]
                else:
                    grading_obj = analysis_result
                
                grading_result = {
                    "chunk_number": chunk_number,
                    "filename": filename,
                    "audio_file": audio_file_basename,
                    "result": grading_obj.model_dump(),
                    "average_score": grading_obj.average_score,
                    "retries": retries_for_file
                }
                all_grading_results.append(grading_result)
                
                # Save the raw response to a file - use audio filename if available
                response_filename = f"{audio_file_basename}_raw_response.json" if audio_file_basename != "unknown_file" else f"chunk_{chunk_number}_raw_response.json"
                with open(os.path.join(params_dir, response_filename), "w") as f:
                    # Replace any non-serializable objects with their string representation
                    json.dump(raw_response_with_metadata, f, indent=2, default=str)
                
                # Create a result dictionary
                result = {
                    "chunk_number": chunk_number,
                    "filename": filename,
                    "audio_file": audio_file_basename,
                    "analysis_result": grading_obj.model_dump(),
                    "average_score": grading_obj.average_score,
                    "analysis_duration_seconds": round(analysis_duration, 2),
                    "retries": retries_for_file
                }
                
                results.append(result)
                successful_files += 1
                
                # Log success
                with open(analysis_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"✅ Analysis completed in {analysis_duration:.2f} seconds\n")
                    if retries_for_file > 0:
                        f.write(f"   Required {retries_for_file} retries to succeed\n")
                    f.write(f"Average score: {grading_obj.average_score}/10\n")
                    f.write(f"Scores: Verbatim={grading_obj.verbatim_match_score}, Preservation={grading_obj.sentence_preservation_score}, "
                            f"Duplication={grading_obj.content_duplication_score}, ContentLoss={grading_obj.content_loss_score}, "
                            f"Transition={grading_obj.join_transition_score}, Flow={grading_obj.contextual_flow_score}\n")
                
                # Save results to files
                result_filename = f"{audio_file_basename}_analysis" if audio_file_basename != "unknown_file" else f"chunk_{chunk_number}_analysis"
                
                # Save as JSON
                with open(os.path.join(analysis_dir, f"{result_filename}.json"), "w") as f:
                    json.dump(result, f, indent=2)
                
                # Save as text
                with open(os.path.join(analysis_dir, f"{result_filename}.txt"), "w") as f:
                    f.write(f"Analysis for {audio_file_basename}\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Filename: {filename}\n")
                    f.write(f"Average Score: {grading_obj.average_score}/10\n")
                    f.write(f"Analysis Duration: {analysis_duration:.2f} seconds\n")
                    if retries_for_file > 0:
                        f.write(f"Required {retries_for_file} retries to succeed\n")
                    f.write("\n")
                    
                    f.write("Grading Results:\n")
                    f.write(f"- Verbatim Match Score: {grading_obj.verbatim_match_score}/10\n")
                    f.write(f"- Sentence Preservation Score: {grading_obj.sentence_preservation_score}/10\n")
                    f.write(f"- Content Duplication Score: {grading_obj.content_duplication_score}/10\n")
                    f.write(f"- Content Loss Score: {grading_obj.content_loss_score}/10\n")
                    f.write(f"- Join Transition Score: {grading_obj.join_transition_score}/10\n")
                    f.write(f"- Contextual Flow Score: {grading_obj.contextual_flow_score}/10\n\n")
                    
                    f.write("Analysis Steps:\n")
                    for step in grading_obj.reasoning_steps:
                        f.write(f"Step {step.step_number}: {step.description}\n")
                        f.write(f"Observation: {step.observation}\n\n")
                    
                    f.write("\nTranscription Content:\n")
                    f.write("-"*50 + "\n")
                    f.write(transcription)
                    f.write("\n" + "-"*50 + "\n")
                
                logger.info(f"Completed analysis for chunk {chunk_number} with score {grading_obj.average_score}/10")
                
            except Exception as e:
                error_msg = f"Error analyzing chunk {chunk_number}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                failed_files += 1
                
                # Log error to the detailed log
                with open(analysis_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"❌ ERROR: {error_msg}\n")
                    f.write(f"Error type: {type(e).__name__}\n")
                    if "ollama" in str(e).lower():
                        f.write("This appears to be an Ollama API error. Check if the Ollama service is running correctly.\n")
                    elif "timeout" in str(e).lower():
                        f.write("This appears to be a timeout error. The model may be taking too long to respond.\n")
                    elif "connection" in str(e).lower():
                        f.write("This appears to be a connection error. Check your network connection and Ollama server status.\n")
                    f.write(f"Retries attempted: {retries_for_file}\n")
                    f.write("\n")
                
                # Add an error entry to our raw responses list
                audio_file_basename = os.path.basename(audio_file) if isinstance(audio_file, str) else str(audio_file)
                error_response = {
                    "chunk_number": chunk_number,
                    "filename": filename,
                    "audio_file": audio_file_basename,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "retries_attempted": retries_for_file
                }
                all_raw_responses.append(error_response)
                
                # Save the error response to a file
                error_filename = f"{audio_file_basename}_error.json" if audio_file_basename != "unknown_file" else f"chunk_{chunk_number}_error.json"
                with open(os.path.join(params_dir, error_filename), "w") as f:
                    json.dump(error_response, f, indent=2)
            
            # Add a separator in the log
            with open(analysis_log_path, 'a', encoding='utf-8') as f:
                f.write("\n" + "-"*50 + "\n\n")
        
        # Save the combined raw responses file for this parameter set
        combined_responses_filename = "all_raw_responses.json"
        with open(os.path.join(params_dir, combined_responses_filename), "w") as f:
            json.dump(all_raw_responses, f, indent=2, default=str)
        
        # Save the combined grading results file for this parameter set
        combined_results_filename = "all_grading_results.json"
        with open(os.path.join(params_dir, combined_results_filename), "w") as f:
            json.dump(all_grading_results, f, indent=2, default=str)
        
        # Create a summary file for this parameter set
        try:
            param_summary = self._create_parameter_summary(
                results, 
                analysis_dir, 
                params_name,
                {
                    "total_files": total_files,
                    "successful_files": successful_files,
                    "failed_files": failed_files,
                    "retried_files": retried_files,
                    "skipped_files": skipped_files
                }
            )
            
            # Add final log entry
            with open(analysis_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\nAnalysis batch completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Parameter set: {params_name}\n")
                f.write(f"Total files: {total_files}\n")
                f.write(f"Successful analyses: {successful_files}\n")
                f.write(f"Failed analyses: {failed_files}\n")
                f.write(f"Files with retries: {retried_files}\n")
                f.write(f"Skipped files: {skipped_files}\n")
                if successful_files > 0:
                    f.write(f"Overall average score: {param_summary['overall_average_score']}/10\n")
                    f.write("\nAverage scores by category:\n")
                    for metric, value in param_summary['detailed_metrics'].items():
                        f.write(f"- {metric.replace('_', ' ').title()}: {value}/10\n")
                
            logger.info(f"Analysis batch completed for parameter set {params_name}. Processed {total_files} files with {successful_files} successful analyses")
            
            # Clean up the working properties
            if hasattr(self, 'current_params_dir'):
                delattr(self, 'current_params_dir')
            if hasattr(self, 'current_chunk_info'):
                delattr(self, 'current_chunk_info')
                
            return param_summary
        except Exception as e:
            logger.error(f"Error creating parameter summary: {str(e)}")
        
        # Clean up the working properties
        if hasattr(self, 'current_params_dir'):
            delattr(self, 'current_params_dir')
        if hasattr(self, 'current_chunk_info'):
            delattr(self, 'current_chunk_info')
        
        return param_summary
    
    async def run_analysis(self, run_dir: str) -> Dict[str, Any]:
        """
        Run analysis on all transcription files in the specified run directory.
        
        Args:
            run_dir: Path to the directory containing transcription files
            
        Returns:
            Dict with analysis results summary
        """
        logger.info(f"Starting analysis of transcriptions in {run_dir}")
        
        # Determine if this is an example run or a real run
        is_example_run = "example" in run_dir.lower()
        
        # For real runs, use AUDIO_AI_OPTIMIZED_DIR/analysis as the output directory
        output_dir = run_dir if is_example_run else AI_ANALYSIS_DIR
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        logger.info(f"Using output directory: {output_dir}")
        
        # Get overlap_seconds from split_state if available
        overlap_seconds = 5  # Default value
        try:
            from split.state import get_split_state
            split_state = get_split_state()
            if split_state and 'settings' in split_state and 'overlap_size' in split_state['settings']:
                overlap_seconds = split_state['settings']['overlap_size']
                logger.info(f"Using overlap_seconds={overlap_seconds} from split_state")
            else:
                logger.warning("Could not find overlap_size in split_state, using default value of 5 seconds")
        except Exception as e:
            logger.warning(f"Error reading split_state: {str(e)}. Using default overlap_seconds=5")
        
        # Initialize progress tracking
        with open(os.path.join(output_dir, "analysis_progress.txt"), "w") as f:
            f.write(f"AI Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("==================================================\n\n")
            f.write(f"Using Ollama model: {self.model_name}\n")
            f.write(f"Analysis run directory: {run_dir}\n")
            f.write(f"Overlap seconds: {overlap_seconds}\n\n")
            
            # Count transcription files
            transcription_files = []
            if os.path.isdir(run_dir):
                transcription_files = [f for f in os.listdir(run_dir) 
                             if not f.startswith('chunk_') and not f.startswith('combined_')]
            
            f.write(f"Found {len(transcription_files)} combinations to analyze:\n")
            for file in transcription_files:
                f.write(f"- {file}\n")
            f.write("\n")
        
        # Process each combination
        results = []
        transcription_files = []
        if os.path.isdir(run_dir):
            transcription_files = [f for f in os.listdir(run_dir) 
                         if not f.startswith('chunk_') and not f.startswith('combined_')]
        
        for i, combo_dir in enumerate(transcription_files):
            combo_name = combo_dir
            
            # Update progress file
            with open(os.path.join(output_dir, "analysis_progress.txt"), "a") as f:
                f.write(f"\nProcessing combination {i+1}/{len(transcription_files)}: {combo_name}\n")
                f.write("--------------------------------------------------\n")
            
            # Check if we have a combined transcription file
            combined_file = os.path.join(run_dir, combo_name, "combined_transcription.json")
            if os.path.exists(combined_file):
                with open(os.path.join(output_dir, "analysis_progress.txt"), "a") as f:
                    f.write(f"  ℹ️ Found combined_transcription.json in main directory: {combined_file}\n")
                    
                try:
                    # Load the transcriptions
                    with open(combined_file, "r") as f:
                        combined_data = json.load(f)
                        
                    transcriptions = combined_data.get("transcriptions", [])
                    
                    if not transcriptions:
                        with open(os.path.join(output_dir, "analysis_progress.txt"), "a") as f:
                            f.write(f"  ⚠️ No transcriptions found in {combined_file}\n")
                        continue
                        
                    with open(os.path.join(output_dir, "analysis_progress.txt"), "a") as f:
                        f.write(f"  ✅ Successfully loaded transcription data with {len(transcriptions)} chunks\n")
                        f.write(f"  📊 Starting analysis of {len(transcriptions)} chunks...\n")
                    
                    # Run the analysis with combo_name as the params_name
                    combo_result = await self.analyze_batch(transcriptions, run_dir, params_name=combo_name, overlap_seconds=overlap_seconds)
                    
                    # Update progress file with results
                    with open(os.path.join(output_dir, "analysis_progress.txt"), "a") as f:
                        f.write(f"  ✅ Completed analysis of {combo_name}\n")
                        f.write(f"    - Total chunks: {combo_result.get('total_files', 0)}\n")
                        f.write(f"    - Successful: {combo_result.get('successful_analyses', 0)}\n")
                        f.write(f"    - Failed: {combo_result.get('failed_files', 0)}\n")
                        if combo_result.get('overall_average_score') is not None:
                            f.write(f"    - Overall score: {combo_result.get('overall_average_score')}/10\n")
                    
                    # Add to results
                    combo_result["combination"] = combo_name
                    results.append(combo_result)
                    
                except Exception as e:
                    logger.error(f"Error analyzing combination {combo_name}: {str(e)}")
                    with open(os.path.join(output_dir, "analysis_progress.txt"), "a") as f:
                        f.write(f"  ❌ Error analyzing combination: {str(e)}\n")
            
            else:
                with open(os.path.join(output_dir, "analysis_progress.txt"), "a") as f:
                    f.write(f"  ⚠️ No combined_transcription.json found in {run_dir}\n")
                    
        # Create overall summary
        total_files = sum(r.get("total_files", 0) for r in results)
        successful_files = sum(r.get("successful_analyses", 0) for r in results)
        failed_files = sum(r.get("failed_files", 0) for r in results)
        
        # Calculate success rate
        success_rate = 0
        if total_files > 0:
            success_rate = (successful_files / total_files) * 100
        
        # Create summary text
        with open(os.path.join(output_dir, "analysis_progress.txt"), "a") as f:
            f.write(f"\n\n✅ Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("==================================================\n\n")
            f.write("SUMMARY:\n")
            
            # Count transcription files
            transcription_files = []
            if os.path.isdir(run_dir):
                transcription_files = [f for f in os.listdir(run_dir) 
                             if not f.startswith('chunk_') and not f.startswith('combined_')]
            
            f.write(f"Total combinations found: {len(transcription_files)}\n")
            f.write(f"Successful combinations: {len(results)}\n")
            f.write(f"Failed combinations: {len(transcription_files) - len(results)}\n")
            f.write(f"Skipped combinations: {0}\n")
            f.write(f"Total files analyzed: {total_files}\n")
            f.write(f"Successful file analyses: {successful_files}\n")
            f.write(f"Failed file analyses: {failed_files}\n")
            f.write(f"Success rate: {success_rate:.1f}%\n\n")
            f.write("To view detailed results, check the 'analysis' folder in each combination directory.\n")
            
        # Create a summary JSON
        summary = {
            "run_dir": run_dir,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "total_combinations": len(transcription_files),
            "successful_combinations": len(results),
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "success_rate": success_rate,
            "combination_results": results
        }
        
        # Save the summary
        with open(os.path.join(output_dir, "analysis_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Analysis completed for run directory: {run_dir}")
        return summary

    def _create_summary(self, results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        """Create a summary of analysis results
        
        Args:
            results: List of analysis results
            output_dir: Directory to store summary
            
        Returns:
            Dictionary with summary information
        """
        logger.info("Creating analysis summary")
        
        # Count successful and failed analyses
        successful_analyses = [r for r in results if "error" not in r]
        failed_analyses = [r for r in results if "error" in r]
        
        # Calculate overall average score
        overall_average_score = 0
        if successful_analyses:
            overall_average_score = sum(r["average_score"] for r in successful_analyses) / len(successful_analyses)
            overall_average_score = round(overall_average_score, 2)
        
        # Create detailed metrics if we have successful analyses
        detailed_metrics = {}
        if successful_analyses:
            try:
                # Get all the metric keys from the first successful analysis
                first_analysis = successful_analyses[0]["analysis_result"]
                metric_keys = [k for k in first_analysis.keys() if isinstance(first_analysis[k], (int, float)) and k != "average_score"]
                
                # Calculate average for each metric
                for key in metric_keys:
                    values = [r["analysis_result"][key] for r in successful_analyses if key in r["analysis_result"]]
                    if values:
                        detailed_metrics[key] = round(sum(values) / len(values), 2)
            except Exception as e:
                logger.warning(f"Error calculating detailed metrics: {str(e)}")
                detailed_metrics = {"error": str(e)}
        
        # Count retried analyses
        retried_files = len([r for r in results if r.get("retries", 0) > 0])
        
        # Create summary dictionary
        summary = {
            "total_files": len(results),
            "successful_analyses": len(successful_analyses),
            "failed_analyses": len(failed_analyses),
            "retried_files": retried_files,
            "skipped_files": 0,  # Not tracked in the legacy method
            "overall_average_score": overall_average_score,
            "detailed_metrics": detailed_metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add error information if there were failures
        if failed_analyses:
            error_types = {}
            for failure in failed_analyses:
                error_type = failure.get("error_type", "Unknown")
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1
            
            summary["error_summary"] = {
                "error_types": error_types,
                "error_examples": [
                    {
                        "chunk_number": failure.get("chunk_number", "Unknown"),
                        "filename": failure.get("filename", "Unknown"),
                        "error": failure.get("error", "Unknown error")
                    }
                    for failure in failed_analyses[:3]  # Include up to 3 examples
                ]
            }
        
        # Save summary to file
        summary_path = os.path.join(output_dir, "analysis", "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Create a human-readable summary
        readable_summary_path = os.path.join(output_dir, "analysis", "summary.txt")
        with open(readable_summary_path, "w") as f:
            f.write("Transcription Analysis Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Generated: {summary['timestamp']}\n\n")
            
            f.write("Overview:\n")
            f.write(f"- Total files analyzed: {summary['total_files']}\n")
            f.write(f"- Successful analyses: {summary['successful_analyses']}\n")
            f.write(f"- Failed analyses: {summary['failed_analyses']}\n")
            f.write(f"- Files with retries: {summary['retried_files']}\n")
            
            if successful_analyses:
                f.write(f"\nOverall Average Score: {overall_average_score}/10\n\n")
                
                f.write("Detailed Metrics:\n")
                for metric, value in detailed_metrics.items():
                    # Format the metric name for better readability
                    formatted_metric = metric.replace("_", " ").title()
                    f.write(f"- {formatted_metric}: {value}/10\n")
            
            if failed_analyses:
                f.write("\nError Summary:\n")
                for error_type, count in summary.get("error_summary", {}).get("error_types", {}).items():
                    f.write(f"- {error_type}: {count} occurrences\n")
                
                f.write("\nExample Errors:\n")
                for i, example in enumerate(summary.get("error_summary", {}).get("error_examples", [])):
                    f.write(f"Example {i+1}:\n")
                    f.write(f"  Chunk: {example.get('chunk_number')}\n")
                    f.write(f"  File: {example.get('filename')}\n")
                    f.write(f"  Error: {example.get('error')}\n\n")
            
            f.write("\nRecommendations:\n")
            if failed_analyses:
                f.write("- Review the error examples and address any issues with the Ollama API or model.\n")
                f.write("- Check if the Ollama service is running correctly.\n")
                f.write("- Verify that the model is available in your Ollama installation.\n")
            
            if successful_analyses:
                # Add recommendations based on the average score
                if overall_average_score < 5:
                    f.write("- The overall quality of transcriptions is low. Consider adjusting transcription parameters.\n")
                elif overall_average_score < 7:
                    f.write("- The transcription quality is moderate. Consider fine-tuning for better results.\n")
                else:
                    f.write("- The transcription quality is good. Continue with current settings.\n")
        
        logger.info(f"Analysis summary created with overall score: {overall_average_score}/10")
        return summary

    def save_prompt_file(self, prompt_text, combination, language_suffix="en"):
        """Save the prompt to a file for debugging purposes."""
        try:
            # Try to get chunk number from the current_chunk_info
            chunk_id = "0"  # Default value
            
            # Try to extract chunk number from the file path if available
            if hasattr(self, 'current_chunk_info') and self.current_chunk_info is not None:
                chunk_path = self.current_chunk_info.get('file_path', '')
                # Check if chunk ID is in the filename
                if chunk_path:
                    # Try to extract the chunk number from the filename (format: chunk_X.txt)
                    chunk_match = re.search(r'chunk_(\d+)\.', chunk_path)
                    if chunk_match:
                        chunk_id = chunk_match.group(1)
            
            # Try to extract the run timestamp from the directory path
            run_timestamp = None
            if hasattr(self, 'current_chunk_info') and self.current_chunk_info is not None:
                chunk_path = self.current_chunk_info.get('file_path', '')
                if chunk_path:
                    # Look for directories that start with "run_" and extract the timestamp
                    path_parts = chunk_path.split('/')
                    for part in path_parts:
                        if part.startswith('run_'):
                            run_timestamp = part.replace('run_', '')
                            break
            
            # Use current time as fallback if no run timestamp found
            if not run_timestamp:
                run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
            # Create the filename with the format: {combination}_chunk{chunk_id}_{language}_{run_timestamp}_prompt.txt
            filename = f"{combination}_chunk{chunk_id}_{language_suffix}_{run_timestamp}_prompt.txt"
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(PROMPT_DIR), exist_ok=True)
            
            # Save the prompt to a file
            prompt_file_path = os.path.join(PROMPT_DIR, filename)
            with open(prompt_file_path, 'w', encoding='utf-8') as file:
                file.write(prompt_text)
            
            logging.warning(f"Saved prompt to file: {prompt_file_path}")
        except Exception as e:
            logging.error(f"Error saving prompt file: {str(e)}") 