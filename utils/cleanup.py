import os
import shutil

def cleanup_python_cache():
    """Remove all Python cache files and directories."""
    for root, dirs, files in os.walk(os.path.dirname(os.path.dirname(__file__))):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_dir)
                print(f"Removed cache directory: {cache_dir}")
            except Exception as e:
                print(f"Error removing {cache_dir}: {e}")
            dirs.remove('__pycache__')
        
        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc'):
                pyc_file = os.path.join(root, file)
                try:
                    os.remove(pyc_file)
                    print(f"Removed cache file: {pyc_file}")
                except Exception as e:
                    print(f"Error removing {pyc_file}: {e}")