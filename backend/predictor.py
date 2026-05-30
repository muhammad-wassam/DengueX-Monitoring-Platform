# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import DepthwiseConv2D

# # 👇 NEW: MobileNetV2 Import (Ye pehchanega k cheez kya hai)
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# # 👇 FIX: Custom Class to handle the version mismatch error (Apka purana fix)
# class CustomDepthwiseConv2D(DepthwiseConv2D):
#     def __init__(self, **kwargs):
#         if 'groups' in kwargs:
#             del kwargs['groups']
#         super().__init__(**kwargs)

# # 1. Path Setup
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "ai_models", "dengue_mosquito_model.h5")

# print(f"Loading Models...")

# # ---------------------------------------------------------
# # LOAD MODEL 1: Gatekeeper (MobileNetV2) - Ye check karega k insect hai ya nahi
# # ---------------------------------------------------------
# try:
#     general_model = MobileNetV2(weights='imagenet')
#     print("✅ Gatekeeper AI (MobileNetV2) Loaded!")
# except Exception as e:
#     print(f"❌ Error loading Gatekeeper: {e}")
#     general_model = None

# # ---------------------------------------------------------
# # LOAD MODEL 2: Apka Custom Dengue Model
# # ---------------------------------------------------------
# try:
#     custom_model = tf.keras.models.load_model(
#         MODEL_PATH, 
#         custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
#     )
#     print("✅ Dengue Custom Model Loaded Successfully!")
# except Exception as e:
#     print(f"❌ Error loading Dengue model: {e}")
#     custom_model = None

# # Settings
# IMAGE_SIZE = (224, 224)
# CONFIDENCE_THRESHOLD_LOW = 70
# CONFIDENCE_THRESHOLD_MED = 80

# CLASS_LABELS = {
#     0: "Dengue Mosquito (Aedes)",
#     1: "Non-Dengue Mosquito"
# }

# # Ye wo list hai jo allow hai (Agar inme se kuch hoga tou hi agay jayega)
# ALLOWED_KEYWORDS = [
#     'mosquito', 'ant', 'bee', 'dragonfly', 
#     'beetle', 'mantis', 'grasshopper', 'cricket', 'arthropod', 'spider',
# ]

# def predict_mosquito(img_path):
#     if custom_model is None or general_model is None:
#         return {"error": "AI Models failed to load. Check server logs."}

#     try:
#         # --- STEP 1: Image Prepare Karna ---
#         img = image.load_img(img_path, target_size=IMAGE_SIZE)
#         img_array = image.img_to_array(img)
        
#         # Copy banayen Gatekeeper k liye (MobileNet alag processing mangta hai)
#         img_gatekeeper = np.expand_dims(img_array.copy(), axis=0)
#         img_gatekeeper = preprocess_input(img_gatekeeper)

#         # Copy banayen Apke Model k liye (Apka model / 255.0 mangta hai)
#         img_custom = np.expand_dims(img_array.copy(), axis=0) / 255.0

#         # --- STEP 2: Gatekeeper Check (Sakhti se mana karna) ---
#         preds = general_model.predict(img_gatekeeper)
#         decoded = decode_predictions(preds, top=5)[0] # Top 5 guesses lo
        
#         # Check karein k top 5 guesses mein koi 'mosquito' ya 'insect' hai?
#         is_insect = False
#         detected_object = decoded[0][1] # Sab se pehla guess kya hai?

#         for _, label, _ in decoded:
#             if any(keyword in label.lower() for keyword in ALLOWED_KEYWORDS):
#                 is_insect = True
#                 break
        
#         # AGAR INSECT NAHI HAI TOU YAHIN ROK DO 🛑
#         if not is_insect:
#             return {
#                 "prediction": "Not a Mosquito",
#                 "confidence": 0,
#                 "warning_level": "INVALID_IMAGE",
#                 "message": f"not mosquito pic (Detected: {detected_object})"
#             }

#         # --- STEP 3: Ab Apka Model Chalega (Kyunke confirm hogya k ye insect hai) ---
#         prediction = custom_model.predict(img_custom)
#         class_index = np.argmax(prediction)
#         confidence = float(np.max(prediction) * 100)

#         if confidence < CONFIDENCE_THRESHOLD_LOW:
#             warning = "LOW_CONFIDENCE"
#         elif confidence < CONFIDENCE_THRESHOLD_MED:
#             warning = "MEDIUM_CONFIDENCE"
#         else:
#             warning = "HIGH_CONFIDENCE"

#         return {
#             "prediction": CLASS_LABELS[class_index],
#             "confidence": round(confidence, 2),
#             "warning_level": warning,
#             "message": "Analysis Complete"
#         }

#     except Exception as e:
#         return {"error": str(e)}