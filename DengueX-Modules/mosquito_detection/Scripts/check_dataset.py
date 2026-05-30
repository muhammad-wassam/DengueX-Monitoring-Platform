import os

root = "F:/Model training/final_dataset"

for cls in os.listdir(root):
    path = os.path.join(root, cls)
    if os.path.isdir(path):
        print(cls, len(os.listdir(path)))