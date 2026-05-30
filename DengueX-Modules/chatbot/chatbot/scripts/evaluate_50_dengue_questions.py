from chatbot_core import chatbot_answer

# =========================
# 50 DENGUE QUESTIONS
# =========================
DENGUE_QUESTIONS = [
    "What is dengue fever?",
    "How does dengue spread?",
    "Which mosquito spreads dengue?",
    "Why is stagnant water dangerous for dengue?",
    "What causes dengue infection?",
    "Is dengue a viral disease?",
    "Where do dengue mosquitoes breed?",
    "Why are dengue cases common in urban areas?",
    "What is the role of Aedes mosquitoes in dengue?",
    "Why does dengue increase after monsoon?",

    "Can dengue spread from person to person?",
    "What environments increase dengue risk?",
    "Why is dengue common in tropical regions?",
    "What type of virus causes dengue?",
    "How do mosquitoes transmit dengue virus?",
    "Why is clean water storage important to prevent dengue?",
    "What is a dengue outbreak?",
    "Why does poor sanitation increase dengue risk?",
    "How does climate affect dengue spread?",
    "What are dengue vectors?",

    "Why are open containers risky for dengue?",
    "What role does rainfall play in dengue spread?",
    "Why is dengue considered a public health issue?",
    "How does urbanization affect dengue?",
    "What is the dengue transmission cycle?",
    "Why do mosquitoes thrive near homes?",
    "What is the incubation role of mosquitoes in dengue?",
    "Why does dengue spread faster in cities?",
    "How do breeding sites contribute to dengue?",
    "Why is mosquito control important for dengue?",

    "What kind of disease is dengue?",
    "Why does dengue affect many countries?",
    "What is vector-borne disease in dengue context?",
    "Why is public awareness important for dengue?",
    "How do environmental conditions affect dengue?",
    "Why is dengue prevention community-based?",
    "What role does water storage play in dengue?",
    "Why is dengue more common in warm climates?",
    "How does population density affect dengue?",
    "Why is dengue monitoring important?",

    "What is the main cause of dengue spread?",
    "Why are dengue mosquitoes active during daytime?",
    "How does lack of sanitation increase dengue?",
    "Why is dengue difficult to control?",
    "What role does human behavior play in dengue spread?",
    "Why is dengue a recurring problem?"
]

# =========================
# SIMPLE KEYWORD CHECK
# =========================
REQUIRED_KEYWORDS = [
    "dengue",
    "mosquito",
    "virus",
    "viral",
    "aedes",
    "water",
    "spread",
    "transmit",
    "vector"
]

def is_answer_reasonable(answer: str) -> bool:
    """
    Conservative correctness check:
    Answer must mention dengue and at least one core concept.
    """
    answer = answer.lower()
    if "dengue" not in answer:
        return False

    for kw in REQUIRED_KEYWORDS:
        if kw in answer:
            return True

    return False

# =========================
# RUN EVALUATION
# =========================
if __name__ == "__main__":
    print("\n========== 50 DENGUE QUESTION EVALUATION ==========\n")

    correct = 0
    wrong = 0

    for idx, question in enumerate(DENGUE_QUESTIONS, 1):
        result = chatbot_answer(question)

        allowed = result["allowed"]
        answer = result["answer"]

        if allowed and is_answer_reasonable(answer):
            status = "RIGHT"
            correct += 1
        else:
            status = "WRONG"
            wrong += 1

        print(f"Q{idx}: {question}")
        print(f"Answer: {answer}")
        print(f"Result: {status}")
        print("-" * 70)

    accuracy = (correct / len(DENGUE_QUESTIONS)) * 100

    print("\n========== FINAL SUMMARY ==========")
    print(f"Total Questions : {len(DENGUE_QUESTIONS)}")
    print(f"Correct         : {correct}")
    print(f"Wrong           : {wrong}")
    print(f"Accuracy        : {accuracy:.2f}%")
    print("==================================\n")
