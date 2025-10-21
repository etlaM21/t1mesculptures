# app_utils.py
import os
import hashlib
from appdirs import user_cache_dir

# Define our app's info
APP_NAME = "T1MESCULPTURES"
APP_AUTHOR = "Malte Hillebrand"

def get_app_cache_dir():
    """Finds or creates the main persistent cache directory for this app."""
    cache_dir = user_cache_dir(APP_NAME, APP_AUTHOR)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def get_project_cache_dir(input_path):
    """
    Creates a unique, persistent cache directory for a specific project
    (based on its input folder path).
    """
    main_cache_dir = get_app_cache_dir()
    
    # Create a unique, stable ID for this project path
    path_hash = hashlib.md5(input_path.encode('utf-8')).hexdigest()
    project_dir = os.path.join(main_cache_dir, path_hash)
    
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        
    return project_dir