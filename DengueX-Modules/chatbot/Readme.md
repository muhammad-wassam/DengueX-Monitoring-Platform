# Overview

This module implements the DengueX AI Chatbot, a hybrid transformer-based dengue awareness assistant designed for public education and safety.

The chatbot provides accurate, non-medical information about dengue fever, including transmission, mosquito breeding, environmental risk factors, seasonal patterns, and public-health awareness.

It is not a diagnostic or medical chatbot.

# Key-Features 

-Transformer-based language model (FLAN-T5)

-Domain-restricted (dengue-related questions only)

-Medical safety guardrails (no diagnosis, treatment, or medication advice)

-Canonical knowledge grounding for critical dengue facts

-Short, clear responses (1–3 sentences)

-Tested using controlled dengue question sets

# What the Chatbot Can Do

-Explain what dengue fever is

-Describe how dengue spreads

-Identify the mosquito responsible

-Explain environmental and seasonal risk factors

-Provide public awareness and prevention guidance

# What the Chatbot Will Not Do

-Diagnose dengue

-Recommend medicines or treatments

-Interpret medical reports or lab results

-Answer non-dengue-related questions

-Unsafe or out-of-scope questions are handled with polite, user-friendly warnings.

# Architecture Summary

-The chatbot uses a hybrid AI architecture:

-Rule-based guardrails for safety and domain restriction

-Canonical knowledge base for verified dengue facts

-Transformer model (FLAN-T5) for controlled natural language generation
