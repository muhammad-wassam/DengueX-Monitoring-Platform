import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 🚀 MODEL & DATA LOADING
MODEL = None
DENGUE_VECTORS = None

# English Response Database for local matching
REPLY_DATABASE = [
    "Common symptoms of dengue include high fever, severe headache, pain behind the eyes, and joint pain. Please consult a professional if you experience these.",
    "To prevent dengue, use mosquito repellents, wear long-sleeved clothing, and ensure there is no stagnant water around your living area.",
    "There is no specific treatment for dengue fever. Patients should stay hydrated, get plenty of rest, and take paracetamol under medical supervision.",
    "Aedes aegypti mosquitoes breed in clean, stagnant water found in containers like old tires, pots, and water tanks. Keeping these areas dry is crucial."
]

def load_logic():
    global MODEL, DENGUE_VECTORS
    try:
        # ⚡ Initializing the fast local model
        MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        anchor_texts = [
            "dengue fever symptoms and signs", 
            "mosquito protection and prevention tips", 
            "platelet count and medical treatment", 
            "aedes aegypti breeding habitat"
        ]
        DENGUE_VECTORS = MODEL.encode(anchor_texts, convert_to_numpy=True)
        print("✅ Local Intelligence Node Active")
    except Exception as e:
        print(f"⚠️ Guard Load Error: {e}")

# Initial load
if MODEL is None:
    load_logic()

def is_dengue_related(query):
    query_lower = query.lower()
    # 🔍 Keywords for quick matching
    essential_parts = ["dengue", "mosq", "fever", "plate", "aede", "sym", "treat"]
    if any(part in query_lower for part in essential_parts):
        return True

    if MODEL is not None:
        query_vec = MODEL.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_vec, DENGUE_VECTORS)[0]
        # Threshold for relevance
        return np.max(similarities) > 0.22
    return False

# 🚀 Main Reply Function used by views.py
def chatbot_reply(query):
    if not is_dengue_related(query):
        return "I am specialized in providing information about Dengue and its prevention. Please ask a related question."

    if MODEL is not None:
        try:
            query_vec = MODEL.encode([query], convert_to_numpy=True)
            similarities = cosine_similarity(query_vec, DENGUE_VECTORS)[0]
            best_match_idx = np.argmax(similarities)
            
            # Select the most relevant response if similarity is high enough
            if similarities[best_match_idx] > 0.3:
                return REPLY_DATABASE[best_match_idx]
        except:
            pass

    return "Dengue is a viral infection transmitted to humans through the bite of infected mosquitoes. Prevention through habitat control is the most effective method."