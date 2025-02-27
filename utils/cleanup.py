import os
import shutil

def cleanup_python_cache():
    """Remove all Python cache files and directories."""
    cache_dirs_removed = 0
    cache_files_removed = 0
    
    for root, dirs, files in os.walk(os.path.dirname(os.path.dirname(__file__))):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_dir)
                cache_dirs_removed += 1
            except Exception as e:
                print(f"Error removing {cache_dir}: {e}")
            dirs.remove('__pycache__')
        
        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc'):
                pyc_file = os.path.join(root, file)
                try:
                    os.remove(pyc_file)
                    cache_files_removed += 1
                except Exception as e:
                    print(f"Error removing {pyc_file}: {e}")
    
    # Print summary message
    if cache_dirs_removed > 0 or cache_files_removed > 0:
        print(f"Cleanup complete: Removed {cache_dirs_removed} cache directories and {cache_files_removed} cache files")
    else:
        print("No cache files found to clean up")