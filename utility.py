import os
import shutil
import datetime
import json

def load_config(config_path="config.json"):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Config file loaded successfully.")
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' deleted.")
    else:
        print(f"File '{file_path}' does not exist.")

def list_files_in_directory(directory_path):
    if os.path.exists(directory_path):
        return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    else:
        print(f"Directory '{directory_path}' does not exist.")
        return []

def copy_file(source_file, destination_file):
    if os.path.exists(source_file):
        shutil.copy(source_file, destination_file)
    else:
        print(f"Source file '{source_file}' does not exist.")

def get_current_timestamp():
    """Returns the current timestamp in the format YYYY-MM-DD HH:MM:SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def file_exists(file_path):
    return os.path.exists(file_path)
