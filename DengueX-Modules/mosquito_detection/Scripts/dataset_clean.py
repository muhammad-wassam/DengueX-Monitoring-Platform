"""
IMPROVED DATASET CLEANING SCRIPT
--------------------------------
Fixes:
✔ Duplicate removal PER CLASS (safe)
✔ Ignores non-image files
✔ More robust handling
"""

import os
import hashlib
from PIL import Image
import cv2

# -------------------------------
# CONFIGURATION
# -------------------------------

DATASET_PATH = r"F:\Model training\final_dataset"

MIN_WIDTH = 100
MIN_HEIGHT = 100

REMOVE_BLUR = False
BLUR_THRESHOLD = 50

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# -------------------------------
# FUNCTIONS
# -------------------------------

def is_corrupt(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return False
    except:
        return True


def is_too_small(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        return width < MIN_WIDTH or height < MIN_HEIGHT
    except:
        return True


def is_blurry(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < BLUR_THRESHOLD
    except:
        return True


def get_image_hash(image_path):
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


# -------------------------------
# MAIN FUNCTION
# -------------------------------

def clean_dataset(root_dir):

    total_removed = 0

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        print(f"\nCleaning class: {class_name}")

        # 🔥 IMPORTANT: reset hash per class
        seen_hashes = set()

        for filename in os.listdir(class_path):

            # Skip non-image files
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue

            file_path = os.path.join(class_path, filename)
            remove_flag = False

            # 1. Corrupt
            if is_corrupt(file_path):
                print(f"Corrupt: {filename}")
                remove_flag = True

            # 2. Too small
            elif is_too_small(file_path):
                print(f"Too small: {filename}")
                remove_flag = True

            # 3. Blur (optional)
            elif REMOVE_BLUR and is_blurry(file_path):
                print(f"Too blurry: {filename}")
                remove_flag = True

            # 4. Duplicate (within same class only)
            else:
                img_hash = get_image_hash(file_path)
                if img_hash in seen_hashes:
                    print(f"Duplicate: {filename}")
                    remove_flag = True
                else:
                    seen_hashes.add(img_hash)

            # Delete if needed
            if remove_flag:
                try:
                    os.remove(file_path)
                    total_removed += 1
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")

    print("\nCleaning completed")
    print(f"Total files removed: {total_removed}")


# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    clean_dataset(DATASET_PATH)