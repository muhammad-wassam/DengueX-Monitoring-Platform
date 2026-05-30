"""
DATASET SPLITTING SCRIPT
------------------------
Purpose:
- Split dataset into training and validation sets
- Maintain class balance
- Prevent data leakage

Output structure:
dataset_split/
    train/
    val/
"""

import os
import shutil
import random

# -------------------------------
# CONFIGURATION
# -------------------------------

# Use forward slashes to avoid Windows path issues
SOURCE_DIR = "F:/Model training/final_dataset"
TRAIN_DIR = "F:/Model training/dataset_split/train"
VAL_DIR = "F:/Model training/dataset_split/val"

# Split ratio (80% train, 20% validation)
SPLIT_RATIO = 0.8

# Set seed for reproducibility (same split every run)
random.seed(42)

# -------------------------------
# FUNCTION: SPLIT ONE CLASS
# -------------------------------

def split_class(class_name):
    """
    Splits one class into train and validation folders.
    """

    src_path = os.path.join(SOURCE_DIR, class_name)

    # Get all image files
    files = os.listdir(src_path)

    # Shuffle for randomness
    random.shuffle(files)

    # Determine split index
    split_index = int(len(files) * SPLIT_RATIO)

    train_files = files[:split_index]
    val_files = files[split_index:]

    # Create destination folders
    train_class_path = os.path.join(TRAIN_DIR, class_name)
    val_class_path = os.path.join(VAL_DIR, class_name)

    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(val_class_path, exist_ok=True)

    # Copy training files
    for f in train_files:
        shutil.copy(
            os.path.join(src_path, f),
            os.path.join(train_class_path, f)
        )

    # Copy validation files
    for f in val_files:
        shutil.copy(
            os.path.join(src_path, f),
            os.path.join(val_class_path, f)
        )

    # Print summary
    print(f"{class_name}  Train: {len(train_files)} | Val: {len(val_files)}")


# -------------------------------
# MAIN EXECUTION
# -------------------------------

if __name__ == "__main__":

    print("Starting dataset split...\n")

    # Loop through each class
    for class_name in os.listdir(SOURCE_DIR):

        class_path = os.path.join(SOURCE_DIR, class_name)

        # Skip non-folder items
        if not os.path.isdir(class_path):
            continue

        split_class(class_name)

    print("\nDataset splitting completed.")