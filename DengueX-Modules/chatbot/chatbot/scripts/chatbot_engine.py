import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from knowledge_base.canonical_answers import CANONICAL_ANSWERS
from knowledge_base.question_classifier import classify_question

# =========================
# MODEL LOADING
# =========================

MODEL_PATH = "model/denguex_flan_t5_final"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

# =========================
# USER-FRIENDLY MESSAGES
# =========================

NON_DENGUE_MESSAGE = (
    "This system is designed to answer questions related to dengue fever only. "
    "Please ask about dengue causes, mosquito transmission, prevention, or public awareness."
)

MEDICAL_BLOCK_MESSAGE = (
    "I cannot help with medical diagnosis, treatment, or medication advice. "
    "Dengue can only be confirmed through medical testing. "
    "Please consult a qualified healthcare professional or visit a government health facility."
)

# =========================
# GUARDRAILS
# =========================

DENGUE_KEYWORDS = [
    "dengue", "aedes", "mosquito", "fever",
    "stagnant", "water", "larvae", "outbreak",
    "vector", "monsoon", "rain"
]

BLOCK_PATTERNS = [
    r"\bdiagnos",
    r"\btreat",
    r"\bmedicine",
    r"\bmedication",
    r"\btablet",
    r"\bdrug",
    r"\bprescrib",
    r"\bcbc",
    r"\bplatelet",
    r"\bmg\b",
    r"\bdo i have dengue",
    r"\bhow to cure",
]

def is_dengue_related(text: str) -> bool:
    text = text.lower()
    return any(word in text for word in DENGUE_KEYWORDS)

def is_medically_blocked(text: str) -> bool:
    text = text.lower()
    return any(re.search(p, text) for p in BLOCK_PATTERNS)

# =========================
# CHATBOT CORE FUNCTION
# =========================

def chatbot_answer(question: str) -> dict:
    # 1. Non-dengue questions
    if not is_dengue_related(question):
        return {
            "allowed": False,
            "answer": NON_DENGUE_MESSAGE
        }

    # 2. Dengue but medically unsafe
    if is_medically_blocked(question):
        return {
            "allowed": False,
            "answer": MEDICAL_BLOCK_MESSAGE
        }

    # 3. Canonical override (critical facts)
    q_type = classify_question(question)

    if q_type in CANONICAL_ANSWERS:
        return {
            "allowed": True,
            "answer": CANONICAL_ANSWERS[q_type]
        }

    # 4. Safe model generation (general awareness only)
    prompt = (
        f"question: {question} "
        f"Answer briefly in 1 to 3 clear sentences for public awareness."
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=90,
            num_beams=3,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "allowed": True,
        "answer": answer.strip()
    }
