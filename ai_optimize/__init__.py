"""AI Optimize module for audio chunking and transcription optimization."""

from .optimizer import AudioOptimizer
import logging

# Define version number
__version__ = "0.1.0"

# Avoid circular imports
def get_create_ai_optimize_tab():
    from .ui import create_ai_optimize_tab
    return create_ai_optimize_tab

# Set up logging
logger = logging.getLogger("ai_optimize")
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)

# Initialize processing times database if it doesn't exist
try:
    from .ui import collect_processing_times_from_existing_files
    processed_count = collect_processing_times_from_existing_files()
    if processed_count:
        logger.info(f"Initialized processing times database with {processed_count} entries")
    else:
        logger.info("No existing analysis files found for processing times database")
except Exception as e:
    logger.error(f"Error initializing processing times database: {str(e)}")

__all__ = ['get_create_ai_optimize_tab', 'AudioOptimizer'] 