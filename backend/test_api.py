import google.generativeai as genai

# --- YAHAN APNI KEY PASTE KAREIN ---
KEY = "YAHAN_APNI_AIzaSy_WALI_KEY_DALEN"

genai.configure(api_key=KEY)

print("Testing connection with Google AI...")

try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Are you working?")
    print("\n✅ SUCCESS! Google AI replied:", response.text)
except Exception as e:
    print("\n❌ FAILED! Error details:")
    print(e)