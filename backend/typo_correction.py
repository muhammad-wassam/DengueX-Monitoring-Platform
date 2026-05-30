from difflib import get_close_matches

COMMON_CORRECTIONS = {
    "dengu": "dengue",
    "denguE": "dengue",
    "dengu fever": "dengue fever",
    "dengu fevr": "dengue fever",
    "dengu symptoms": "dengue symptoms",
    "mosqito": "mosquito",
    "mosquto": "mosquito",
    "mosquitoes": "mosquito",
    "platelets": "platelet",
    "plaetlet": "platelet",
    "ns_1": "ns1",
    "ns-1": "ns1",
}

KEYWORDS = [
    "dengue","dengue fever","symptoms","mosquito","aedes",
    "platelet","rash","ns1","serology","bleeding",
    "fever","vomiting","abdominal pain","hospital"
]

def correct_typo(text: str) -> str:
    if not text:
        return text
    t = text.strip()

    lowered = t.lower()

    # Dictionary replacements
    for k, v in COMMON_CORRECTIONS.items():
        if k in lowered:
            lowered = lowered.replace(k, v)

    # Fuzzy token correction
    words = lowered.split()
    repaired = []
    for w in words:
        if w in KEYWORDS or len(w) <= 2:
            repaired.append(w)
            continue

        close = get_close_matches(w, KEYWORDS, n=1, cutoff=0.85)
        if close:
            repaired.append(close[0])
        else:
            repaired.append(w)

    return " ".join(repaired)
