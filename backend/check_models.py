import google.generativeai as genai

# --- APNI API KEY YAHAN DALEIN ---
GOOGLE_API_KEY = "AIzaSyAphBN5nKZinHOGxBZS0Hu-qQqhI6VeISM"
genai.configure(api_key=GOOGLE_API_KEY)

print("Fetching available models...")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print("Error:", e)