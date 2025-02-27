#!/usr/bin/env python3
import os
import subprocess
import sys
import platform
from utils.logger import logger

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Git is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Git is not installed. Please install Git and try again.")
        return False
    
    try:
        subprocess.run(["make", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Make is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Make is not installed. Please install Make and try again.")
        return False
    
    return True

def clone_whisper_cpp():
    """Clone the whisper.cpp repository."""
    if os.path.exists("whisper.cpp"):
        logger.info("whisper.cpp directory already exists")
        return True
    
    try:
        logger.info("Cloning whisper.cpp repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/whisper.cpp.git"],
            check=True
        )
        logger.info("whisper.cpp repository cloned successfully")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to clone whisper.cpp repository: {e}")
        return False

def compile_whisper_cpp():
    """Compile whisper.cpp."""
    try:
        logger.info("Compiling whisper.cpp...")
        os.chdir("whisper.cpp")
        
        # Determine make command based on platform and CPU
        make_cmd = ["make"]
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            # Check if Apple Silicon
            if platform.machine() == "arm64":
                make_cmd.append("WHISPER_COREML=1")
        
        subprocess.run(make_cmd, check=True)
        logger.info("whisper.cpp compiled successfully")
        os.chdir("..")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to compile whisper.cpp: {e}")
        os.chdir("..")
        return False

def download_model():
    """Download the large-v3-turbo model."""
    model_dir = os.path.join("whisper.cpp", "models")
    model_path = os.path.join(model_dir, "ggml-large-v3-turbo.bin")
    
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        return True
    
    try:
        logger.info("Downloading large-v3-turbo model...")
        os.chdir("whisper.cpp")
        subprocess.run(
            ["bash", "models/download-ggml-model.sh", "large-v3-turbo"],
            check=True
        )
        logger.info("Model downloaded successfully")
        os.chdir("..")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to download model: {e}")
        os.chdir("..")
        return False

def main():
    """Main function to set up whisper.cpp."""
    logger.info("Setting up whisper.cpp...")
    
    if not check_dependencies():
        logger.error("Missing dependencies. Please install them and try again.")
        return False
    
    if not clone_whisper_cpp():
        logger.error("Failed to clone whisper.cpp repository.")
        return False
    
    if not compile_whisper_cpp():
        logger.error("Failed to compile whisper.cpp.")
        return False
    
    if not download_model():
        logger.error("Failed to download model.")
        return False
    
    logger.info("whisper.cpp setup completed successfully!")
    logger.info("You can now use the transcription functionality in the application.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 