def classify_question(question: str) -> str:
    q = question.lower()

    # --- Core definitions ---
    if "what is dengue" in q or "define dengue" in q:
        return "what_is_dengue"

    # --- Transmission ---
    if "spread" in q or "transmit" in q:
        return "how_dengue_spreads"

    if "person to person" in q:
        return "person_to_person"

    # --- Mosquito specific ---
    if "which mosquito" in q or "aedes" in q:
        return "which_mosquito"

    # --- Breeding & water ---
    if "breed" in q or "breeding" in q or "stagnant water" in q:
        return "breeding_sites"

    # --- Seasonal / climate ---
    if "monsoon" in q or "rain" in q or "seasonal" in q:
        return "why_after_monsoon"

    if "summer" in q:
        return "summer_rise"

    if "tropical" in q:
        return "tropical_regions"

    if "climate" in q:
        return "climate_effect"

    # --- Urban / public health ---
    if "urban" in q or "city" in q:
        return "urban_risk"

    if "public health" in q:
        return "public_health_problem"

    if "control" in q or "difficult" in q:
        return "urban_control_difficulty"

    if "vector-borne" in q:
        return "vector_borne"

    # --- Community & prevention ---
    if "community" in q or "reduce risk" in q:
        return "community_risk_reduction"

    if "waste" in q:
        return "waste_management"

    if "clean water" in q:
        return "clean_water_storage"

    if "cover" in q and "container" in q:
        return "cover_containers"

    # --- Fallback ---
    return "general"
