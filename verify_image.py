import os
from PIL import Image

def check_image_integrity(file_path):
    """Verifies the integrity of an image file using Pillow."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        with Image.open(file_path) as img:
            img.verify() # Checks for broken data
            print(f"Success: '{file_path}' is a valid image.")
            print(f"Detected Format: {img.format}")
    except Exception as e:
        print(f"Error: The image '{file_path}' appears to be corrupted.\nDetails: {e}")

if __name__ == "__main__":
    # Checking the file provided in context
    target_image = r"c:\code\authen_check\test_image.jpg"
    check_image_integrity(target_image)