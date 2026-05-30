import json
import os
import random
from tqdm import tqdm

# File Paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "raw", "dengue_qa_train.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset", "expanded", "dengue_qa_train_expanded.jsonl")

 # Output Size

TARGET_SIZE = 2000

TEMPLATES = [
    "question: {}?",
    "question: How does {}?",
    "question: In what way does {}?",
    "question: Why does {}?",
    "question: What causes {}?",
    "question: How can {} be explained?",
    "question: What is the reason behind {}?"
]

def normalize(q):
    q = q.replace("question:", "").strip()
    return q.rstrip("?")

# Load  dataset

records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} seed QA pairs")

expanded = []
seen = set()

print("Expanding dataset...")
for rec in tqdm(records):
    base_q = normalize(rec["input"])

    for tpl in TEMPLATES:
        new_q = tpl.format(base_q)

        if new_q not in seen:
            expanded.append({
                "input": new_q,
                "output": rec["output"]
            })
            seen.add(new_q)

        if len(expanded) >= TARGET_SIZE:
            break

    if len(expanded) >= TARGET_SIZE:
        break

random.shuffle(expanded)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for r in expanded:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Done. Final dataset size: {len(expanded)}")
