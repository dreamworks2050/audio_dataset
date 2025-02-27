import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Define color schemes for different log levels
COLOR_SCHEME = {
    'DEBUG': Fore.CYAN,
    'INFO': Fore.GREEN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'CRITICAL': Fore.RED + Style.BRIGHT
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output"""
    def format(self, record):
        # Save original levelname
        levelname = record.levelname
        # Apply color if level has one defined
        if levelname in COLOR_SCHEME:
            record.levelname = f"{COLOR_SCHEME[levelname]}{levelname}{Style.RESET_ALL}"
        # Format the message
        message = super().format(record)
        # Restore original levelname
        record.levelname = levelname
        return message

class Logger:
    def __init__(self, name='audio_dataset'):
        # Create logs directory if it doesn't exist
        self.logs_dir = 'logs'
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Store the log file path
        self.log_file = os.path.join(self.logs_dir, f'{name}.log')
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def clear_logs(self):
        """Clear the log file by truncating it to zero size."""
        try:
            # Open the file in write mode, which truncates it
            with open(self.log_file, 'w') as f:
                pass
            self.info(f"Log file cleared: {self.log_file}")
            return True
        except Exception as e:
            self.error(f"Failed to clear log file: {str(e)}")
            return False
    
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)

# Create a global logger instance
logger = Logger()