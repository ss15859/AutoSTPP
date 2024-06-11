import os
import sys

def create_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_path}' created successfully.")
    except Exception as e:
        print(f"Failed to create directory '{directory_path}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_directory.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    create_directory(directory_path)
