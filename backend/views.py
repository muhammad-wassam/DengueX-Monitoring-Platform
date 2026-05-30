import keras
import re
import uuid
import random
import io
import pathlib
from django.shortcuts import get_object_or_404
import json
import os
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
# Third-party Imports
from PIL import Image
import requests
import torch
import torchvision.transforms as T
from .chatbot_logic import is_dengue_related
from .feature1_chatbot.wrapper import chatbot_reply as feature_chatbot_reply
import traceback

# Django Core Imports
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.shortcuts import render
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate
from .chatbot_logic import chatbot_reply

# REST Framework Imports
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from .serializers import MosquitoReportSerializer
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.core.mail import send_mail
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
import base64

# Local Models (All grouped together)
from .models import (
    MosquitoReport,
    UserProfile, 
    DengueStat, 
    HealthTip, 
    ChatSession, 
    ChatMessage, 
    NewsPost,
    OTPRecord,
)

import time
import traceback
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from .chatbot_logic import is_dengue_related
from .models import ChatSession, ChatMessage
import os
from django.conf import settings
from PIL import Image
from fastai.learner import load_learner
   

# 🚀 MOBILE NET LOADING LOGIC
MODEL_DIR = os.path.join(settings.BASE_DIR, 'chatbot', 'ai_models')
FASTAI_MODEL_PATH = os.path.join(MODEL_DIR, 'mosquito_classifier.pkl.zip')
model = None
labels = []
fastai_preprocess = None

def load_local_model():
    global model, labels, fastai_preprocess
    try:
        if os.path.exists(FASTAI_MODEL_PATH):
            # FastAI models trained on Linux can store PosixPath in pickle.
            pathlib.PosixPath = pathlib.WindowsPath
            learner = load_learner(FASTAI_MODEL_PATH, cpu=True)
            model = learner.model.eval()
            labels = [str(v) for v in getattr(learner.dls, 'vocab', [])]
            fastai_preprocess = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print(f"✅ SUCCESS: FastAI model loaded ({FASTAI_MODEL_PATH})")
            return
    except Exception as e:
        print(f"⚠️ FastAI Load Error: {e}")

    try:
        # 1. Architecture load karein (config.json)
        with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
            model_config = json.load(f) # JSON object read karein
        
        # Keras 3 native loading
        model = keras.models.model_from_json(json.dumps(model_config))

        # 2. Weights load karein (model.weights.h5)
        model.load_weights(os.path.join(MODEL_DIR, 'model.weights.h5'))
        
        # 3. Labels load karein (metadata.json se labels uthayein)
        with open(os.path.join(MODEL_DIR, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            # Aapke metadata.json mein labels isi order mein hain
            labels = metadata.get('labels', ["Aedes", "Culex", "Anopheles", "Culiseta", "Toxorhynchites", "Psorophora"])
            
        print("✅ SUCCESS: MobileNet Model & Weights Loaded (Keras 3)")
    except Exception as e:
        print(f"❌ Keras Load Error: {e}")

# Server start hotay hi load ho jaye ga
load_local_model()

# ==========================================
# 🔬 LAB DETECTION FUNCTION
# ==========================================
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([AllowAny])
def detect_mosquito_specie(request):
    global model, labels, fastai_preprocess
    try:
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"result": "ERROR", "message": "Specimen missing."}, status=200)

        if model is None:
            return Response({"result": "ERROR", "message": "AI Node not active."}, status=200)

        raw_img = Image.open(image_file).convert('RGB')

        # 1) FastAI-trained PyTorch path (new integrated model)
        if fastai_preprocess is not None and labels:
            input_tensor = fastai_preprocess(raw_img).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
            class_idx = int(torch.argmax(probs).item())
            identified_specie = labels[class_idx]
            confidence = float(probs[class_idx].item()) * 100
        else:
            # 2) Keras fallback path
            img = raw_img.resize((224, 224))
            img_array = np.array(img) / 255.0  # Normalization [0, 1]
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100
            identified_specie = labels[class_idx]

        identified_specie_normalized = identified_specie.lower().replace("-", "_")
        is_dengue = identified_specie_normalized == "aedes"

        # Risk & Habitat logic (Based on species)
        risk_map = {
            "Aedes": "HIGH (Dengue Vector)",
            "Culex": "MODERATE (West Nile/Filariasis)",
            "Anopheles": "HIGH (Malaria Vector)",
            "Non_Aedes": "LOW (Not classified as Aedes dengue vector)",
            "Unknown": "LOW (Unable to classify confidently)",
        }
        habitat_map = {
            "Aedes": "Stagnant water in containers, pots, and tires.",
            "Culex": "Polluted water, drains, and ditches.",
            "Anopheles": "Clean, slow-moving fresh water.",
            "Non_Aedes": "General outdoor breeding areas; remove standing water.",
            "Unknown": "Image unclear. Capture a closer and well-lit specimen image.",
        }

        return Response({
            "result": identified_specie.upper(),
            "common_name": f"{identified_specie} Mosquito",
            "specie": identified_specie,
            "is_dengue": is_dengue,
            "risk": risk_map.get(identified_specie, "LOW"),
            "habitat": habitat_map.get(identified_specie, "Natural surroundings"),
            "message": f"Identified {identified_specie} with {confidence:.1f}% confidence.",
            "confidence": f"{confidence:.1f}%"
        }, status=200)

    except Exception as e:
        print(f"🚨 Prediction Error: {e}")
        return Response({"result": "ERROR", "message": str(e)}, status=200) 

    
@api_view(['POST'])
@permission_classes([AllowAny])
def chatbot_response(request):
    user_message = request.data.get('message', '').strip()
    session_id = request.data.get('session_id')

    if not user_message:
        return Response({"error": "Message is required"}, status=400)

    # We route all messages through the better Dengue retrieval engine
    # (feature1_chatbot) which does its own relevance + urgency detection.
    try:
        result = feature_chatbot_reply(user_message)
        bot_reply = result.get("reply", "")
        intent = result.get("urgency", "non-urgent")
        confidence = result.get("confidence", 0.0)
    except Exception as e:
        bot_reply = "I'm having trouble processing that locally. Please try again."
        intent = "error"
        confidence = 0.0

    # History Saving
    current_session_id = session_id
    if request.user.is_authenticated:
        session = None
        if session_id:
            session = ChatSession.objects.filter(id=session_id, user=request.user).first()
        if not session:
            session = ChatSession.objects.create(user=request.user, title=user_message[:30])
            current_session_id = session.id
        
        ChatMessage.objects.create(session=session, role='user', content=user_message)
        ChatMessage.objects.create(session=session, role='assistant', content=bot_reply)

    return Response({
        "response": bot_reply,
        "intent": intent,
        "confidence": confidence,
        "session_id": current_session_id
    })

# --- 📂 3. SESSION MANAGEMENT (Fixed) ---
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_chat_sessions(request):
    sessions = ChatSession.objects.filter(user=request.user).order_by('-created_at')
    return Response([{"id": s.id, "title": s.title} for s in sessions])

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_chat_messages(request, session_id):
    try:
        session = ChatSession.objects.get(id=session_id, user=request.user)
        messages = session.messages.all().order_by('timestamp')
        # FIX: Returning 'sender' as 'bot' or 'user' for frontend mapping
        return Response([{"sender": 'bot' if m.role == 'assistant' else 'user', "text": m.content} for m in messages])
    except:
        return Response({"error": "Chat not found"}, status=404)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_chat_session(request, session_id):
    ChatSession.objects.filter(id=session_id, user=request.user).delete()
    return Response({"message": "Deleted"}, status=200)

# views.py
@api_view(['GET'])
def get_report_details(request, pk):
    # pk ka matlab hai Primary Key (Unique ID)
    report = MosquitoReport.objects.get(pk=pk) 

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def submit_report_api(request):
    try:
        # Frontend se data uthayen
        desc = request.data.get('description', '')
        area = request.data.get('area_name', 'Unknown')
        lat = request.data.get('latitude')
        lng = request.data.get('longitude')
        img = request.FILES.get('image')

        report = MosquitoReport.objects.create(
            user=request.user,
            image=img,
            description=desc,
            area_name=area,
            latitude=float(lat) if lat and lat != "" else None,
            longitude=float(lng) if lng and lng != "" else None
        )

        return Response({"message": "Log Created", "id": report.id}, status=201)

    except Exception as e:
        print(f"🚨 Submit Error: {str(e)}") # Terminal check karein
        return Response({"error": str(e)}, status=400)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_reports_api(request):
    # Sirf is user ki reports laao
    reports = MosquitoReport.objects.filter(user=request.user).order_by('-created_at')
    data = [{
        "id": r.id, 
        "area_name": r.area_name, 
        "status": r.status,
        "image": r.image.url if r.image else None,
        "description": r.description,
        "latitude": r.latitude,
        "longitude": r.longitude,
        "created_at": r.created_at
    } for r in reports]
    return Response(data)

# chatbot/views.py

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_report(request, pk):
    try:
        # 1. Pehle pure database mein ye ID dhoondo
        report = MosquitoReport.objects.filter(pk=pk).first()
        
        if not report:
            return Response({"error": "Report not found"}, status=404)

        # 2. Force Delete: Chahay user match ho ya na ho (FYP Testing Fix)
        report.delete()
        
        print(f"🔥 Force Deleted Report ID: {pk}")
        return Response({"message": "Successfully erased from core."}, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_all_reports(request):
    # ⚡ MASTER FORMAT: Sab saaf
    MosquitoReport.objects.filter(user=request.user).delete()
    return Response({"message": "Core Formatted"}, status=200)

@api_view(['GET'])
@permission_classes([IsAdminUser]) # Taake sirf admin dekh sakay
def admin_get_all_reports(request):
    # 🚨 Ensure karein ke yahan MosquitoReport model use ho raha hai
    reports = MosquitoReport.objects.all().order_by('-created_at')
    serializer = MosquitoReportSerializer(reports, many=True)
    
    # Stats calculate karke bhejein taake boxes update hon
    return Response({
        "total": reports.count(),
        "pending": reports.filter(status='pending').count(),
        "resolved": reports.filter(status='resolved').count(),
        "rejected": reports.filter(status='rejected').count(),
        "reports": serializer.data
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_profile_image(request):
    try:
        # Aapke model ka naam 'UserProfile' hai
        from .models import UserProfile 
        
        # User ka profile nikalne ka sahi tareeqa
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        
        if 'profile_image' in request.FILES:
            profile.profile_image = request.FILES['profile_image']
            profile.save()
            
            # Image ka full URL return karna taake React foran dikha sakay
            image_url = request.build_absolute_uri(profile.profile_image.url)
            return Response({
                "profile_image": image_url, 
                "message": "Success! Image updated."
            }, status=200)
        
        return Response({"error": "No image found in request"}, status=400)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
@permission_classes([AllowAny])
def login_api(request):
    username = request.data.get("username")
    password = request.data.get("password")

    user = authenticate(username=username, password=password)

    if user is not None:
        token, _ = Token.objects.get_or_create(user=user)
        return Response({
            "message": "Login successful", 
            "username": user.username,
            "token": token.key,
            "is_admin": user.is_staff  # ✅ YE HAI WO LINE JO MISSING THI
        })
    else:
        return Response({"error": "Invalid username or password"}, status=401)
    
@api_view(['POST'])
@permission_classes([IsAdminUser]) # // Crucial: Only Admins can access this:
def admin_change_password_api(request):
    try:
        user_id = request.data.get('user_id')
        new_password = request.data.get('new_password')

        # 1. Validation
        if not user_id or not new_password:
            return Response({"error": "User ID and New Password are required."}, status=400)

        if len(new_password) < 6:
            return Response({"error": "Password must be at least 6 characters long."}, status=400)

        # 2. Get User
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found."}, status=404)

        # 3. Set New Password (Hashing)
        user.set_password(new_password) 
        user.save()

        return Response({"message": f"Password for {user.username} updated successfully!"}, status=200)

    except Exception as e:
        print("Admin Password Change Error:", e)
        return Response({"error": str(e)}, status=500)    

@api_view(['POST', 'GET'])
@permission_classes([IsAuthenticated])
def get_profile(request):
    try:
        user = request.user
        # Agar user ka profile nahi bana hua tou naya bana do
        profile, created = UserProfile.objects.get_or_create(user=user)

        data = {
            "username": user.username,
            "email": user.email, # Email main User model se aayegi
            "full_name": profile.full_name,
            "age": profile.age,
            "blood_group": profile.blood_group,
            "city": profile.city,
            "emergency_contact": profile.emergency_contact,
            "previous_infection": profile.previous_infection,
            "comorbidities": profile.comorbidities,
            "travel_history": profile.travel_history,
        }
        return Response(data)
    except Exception as e:
        print("Get Profile Error:", str(e))
        return Response({"error": str(e)}, status=400)
    
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_profile_api(request):
    try:
        user = request.user
        profile = UserProfile.objects.get(user=user)

        profile.full_name = request.data.get("full_name", profile.full_name)
        if request.data.get("age"): profile.age = int(request.data.get("age"))
        profile.blood_group = request.data.get("blood_group", profile.blood_group)
        profile.is_vaccinated = request.data.get("is_vaccinated", profile.is_vaccinated)
        profile.recent_test_date = request.data.get("recent_test_date", profile.recent_test_date)
        
        profile.save()
        return Response({"message": "Profile updated successfully!"})
        
    except UserProfile.DoesNotExist:
        return Response({"error": "Profile not found"}, status=404)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    try:
        user = request.user
        profile, created = UserProfile.objects.get_or_create(user=user)
        data = request.data

        # 1. Base User model mein email save karein
        if 'email' in data and data['email'].strip() != "":
            user.email = data['email']
            user.save()

        # 2. Age ke maslay ko fix karna (Agar khali ho tou None save karo)
        age_val = data.get('age')
        if age_val == "":
            profile.age = None
        elif age_val is not None:
            profile.age = age_val

        # 3. Baki fields save karein
        profile.full_name = data.get('full_name', profile.full_name)
        profile.blood_group = data.get('blood_group', profile.blood_group)
        profile.city = data.get('city', profile.city)
        profile.emergency_contact = data.get('emergency_contact', profile.emergency_contact)
        profile.previous_infection = data.get('previous_infection', profile.previous_infection)
        profile.comorbidities = data.get('comorbidities', profile.comorbidities)
        profile.travel_history = data.get('travel_history', profile.travel_history)

        profile.save()
        return Response({"message": "Profile updated successfully!"})
    
    except Exception as e:
        print("❌ PROFILE UPDATE ERROR:", str(e)) # Terminal mein masla print hoga
        return Response({"error": "Backend Error, check terminal"}, status=400)

# ==========================================
# 4. CHATBOT APIs (DEBUG MODE ENABLED)
# ==========================================

@api_view(["POST"])
@permission_classes([AllowAny])
def signup_api(request):
    """Clean Signup: No security questions, only Email for OTP recovery"""
    username = request.data.get("username")
    password = request.data.get("password")
    email = request.data.get("email") # Recovery ke liye email zaroori hai

    if not username or not password or not email:
        return Response({"error": "Username, Password and Email are required."}, status=400)

    if User.objects.filter(username=username).exists():
        return Response({"error": "Username is already taken."}, status=400)

    try:
        user = User.objects.create_user(username=username, password=password, email=email)
        UserProfile.objects.create(user=user, email=email) # Profile mein email save
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"message": "Account created successfully!", "token": token.key, "username": user.username})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
@permission_classes([AllowAny])
def google_login_api(request):
    email = request.data.get("email")
    name = request.data.get("name")
    if not email: return Response({"error": "Email is required"}, status=400)
    user = User.objects.filter(username=email).first()
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"message": "Login Successful", "username": user.username, "token": token.key})
    try:
        random_password = str(uuid.uuid4()) 
        user = User.objects.create_user(username=email, email=email, password=random_password)
        UserProfile.objects.create(user=user, full_name=name, security_question="Google", security_answer="Google")
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"message": "Account Created", "username": user.username, "token": token.key})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_profile_api(request):
    try:
        profile = UserProfile.objects.get(user=request.user)
        profile.full_name = request.data.get("full_name", profile.full_name)
        if request.data.get("age"): profile.age = int(request.data.get("age"))
        profile.blood_group = request.data.get("blood_group", profile.blood_group)
        profile.is_vaccinated = request.data.get("is_vaccinated", profile.is_vaccinated)
        profile.recent_test_date = request.data.get("recent_test_date", profile.recent_test_date)
        profile.save()
        return Response({"message": "Updated!"})
    except: return Response({"error": "Error updating"}, status=404)

@api_view(['GET'])
@permission_classes([AllowAny])
def dashboard_stats_api(request):
    try:
        # 1. Calculate Aggregate Stats
        stats = DengueStat.objects.aggregate(
            total_cases=Sum('active_cases') + Sum('recovered') + Sum('deaths'),
            active=Sum('active_cases'),
            recovered=Sum('recovered'),
            deaths=Sum('deaths')
        )

        # 2. Get All City Stats (For Map & Table)
        city_stats = DengueStat.objects.all()
        city_data = [
            {
                "id": stat.id,
                "city": stat.city_name,
                "active": stat.active_cases,
                "recovered": stat.recovered,
                "deaths": stat.deaths,
                "latitude": stat.latitude,
                "longitude": stat.longitude
            } 
            for stat in city_stats
        ]

        tips = HealthTip.objects.all().order_by('-date_posted')[:5]
        tips_data = [
            {
                "id": t.id, 
                "title": t.title, 
                "description": t.description,
                "date": t.date_posted.strftime("%Y-%m-%d") if t.date_posted else ""
            } 
            for t in tips
        ]

        # 4. Final Response Structure
        return Response({
            "summary": {
                "total_reports": stats['total_cases'] or 0,
                "active": stats['active'] or 0,
                "recovered": stats['recovered'] or 0,
                "deaths": stats['deaths'] or 0
            },
            "stats": city_data,  # Map aur Cards ke liye
            "city_stats": city_data, # Backward compatibility ke liye
            "health_tips": tips_data
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

# ✅ 2. DELETE TIP (POST Method Use Karein)
@api_view(['POST'])  # 👈 Ye Line Sabse Zaroori Hai
@permission_classes([IsAdminUser])
def admin_delete_tip(request):
    try:
        tip_id = request.data.get('id')
        
        if not tip_id:
            return Response({"error": "Tip ID is required"}, status=400)
            
        deleted, _ = HealthTip.objects.filter(id=tip_id).delete()
        
        if deleted:
            return Response({"message": "Tip deleted successfully!"})
        else:
            return Response({"error": "Tip not found"}, status=404)

    except Exception as e:
        return Response({"error": str(e)}, status=500)
# ✅ 2. DELETE HEALTH TIP

@api_view(["GET"])
@permission_classes([AllowAny])
def analytics_data_api(request):
    try:
        total_active_admin = DengueStat.objects.aggregate(Sum('active_cases'))['active_cases__sum'] or 0

        # 2. Chart Data Generate karein
        chart_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=29) # Pichle 30 din
        
        # Din count karein
        delta = (end_date - start_date).days + 1

        for i in range(delta):
            date_obj = start_date + timedelta(days=i)
            display_date = date_obj.strftime("%b %d") # e.g. "Dec 23"
            days_from_today = (end_date.date() - date_obj.date()).days # 0 means Aaj, 1 means Kal...
            
            if days_from_today == 0:
                # Aaj ka data = Total Admin Active Cases
                estimated = total_active_admin
            else:
                decrease_amount = int(total_active_admin * 0.05 * days_from_today) 
                # Ya simple minus logic: (days * 10)
                
                estimated = max(0, total_active_admin - decrease_amount)

            chart_data.append({
                "date": display_date,
                "cases": estimated
            })
            
        return Response(chart_data)

    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
@api_view(["GET"])
@permission_classes([AllowAny])
def get_news_api(request):
    """Public aur Admin News List ke liye"""
    # 🚀 FIX: '-date' ko badal kar '-date_posted' karein
    news_list = NewsPost.objects.all().order_by('-date_posted')
    data = [{
        "id": n.id, 
        "title": n.title, 
        "city": n.city, 
        "content": n.content, 
        # 🚀 FIX: n.date ko n.date_posted karein
        "date": n.date_posted.strftime("%d %b, %Y") if n.date_posted else "N/A"
    } for n in news_list]
    return Response(data)

@api_view(['POST'])
@permission_classes([IsAdminUser])  # Sirf Admin hi ye kar sake
def toggle_user_block_status(request):
    try:
        user_id = request.data.get('user_id')
        user = User.objects.get(id=user_id)
        # Admin khud ko block na kar sake
        if user.is_superuser:
            return Response({"error": "Cannot block Super Admin"}, status=400)

        # Status Toggle (Agar True hai to False, False hai to True)
        user.is_active = not user.is_active
        user.save()
        
        status_msg = "Unblocked" if user.is_active else "Blocked"
        return Response({"message": f"User {status_msg} successfully", "is_active": user.is_active})

    except User.DoesNotExist:
        return Response({"error": "User not found"}, status=404)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
@permission_classes([IsAdminUser])
def admin_update_report_status(request):
    """Admin status badal sake: Resolved, Rejected, Pending"""
    try:
        report_id = request.data.get('id')
        new_status = request.data.get('status').lower() # 'resolved', 'rejected', etc.

        # 🚀 MosquitoReport model use ho raha hai
        report = MosquitoReport.objects.get(id=report_id)
        report.status = new_status
        report.save()
        
        return Response({"message": f"Status updated to {new_status}"}, status=200)
    except MosquitoReport.DoesNotExist:
        return Response({"error": "Report not found"}, status=404)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
@permission_classes([IsAdminUser])
def admin_update_stats(request):
    try:
        # Raw Data
        raw_city = request.data.get('city_name')
        
        if not raw_city:
            return Response({"error": "City Name is required"}, status=400)

        # ✨ MAGIC FIX: Naam ko Standardize karein (e.g., "multan " -> "Multan")
        city_name = raw_city.strip().title()

        active = request.data.get('active_cases')
        recovered = request.data.get('recovered')
        deaths = request.data.get('deaths')
        lat = request.data.get('latitude')
        lon = request.data.get('longitude')

        defaults_data = {
            'active_cases': int(active) if active else 0,
            'recovered': int(recovered) if recovered else 0,
            'deaths': int(deaths) if deaths else 0
        }

        # Location tab hi update karein agar user ne di ho
        if lat and lon:
            defaults_data['latitude'] = float(lat)
            defaults_data['longitude'] = float(lon)

        # update_or_create ab "Multan" dhoonde ga, chahe user ne "multan" likha ho
        obj, created = DengueStat.objects.update_or_create(
            city_name=city_name, 
            defaults=defaults_data
        )

        return Response({"message": f"{'Created' if created else 'Updated'} {city_name} successfully!"})

    except Exception as e:
        return Response({"error": str(e)}, status=500)

# ✅ 2. DELETE CITY API (NEW)
@api_view(['POST'])
@permission_classes([IsAdminUser])
def admin_delete_city(request):
    try:
        city_id = request.data.get('id')
        if not city_id:
            return Response({"error": "ID required"}, status=400)
            
        DengueStat.objects.filter(id=city_id).delete()
        
        return Response({"message": "City deleted successfully!"})
    except Exception as e:
        
        return Response({"error": str(e)}, status=500)
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
 # Dono models import kar lein

@api_view(['POST'])  # 👈 Sirf POST hona chahiye
@permission_classes([IsAdminUser])
def admin_delete_news(request):
    try:
        news_id = request.data.get('id')
        print(f"🗑️ Delete Request received for ID: {news_id}")

        if not news_id:
            return Response({"error": "News ID is required"}, status=400)
            
        # Pehle NewsPost (Standard) check karein
        deleted, _ = NewsPost.objects.filter(id=news_id).delete()
        # Agar wahan nahi mila, to NewsUpdate (Backup) check karein
        if not deleted:
            deleted, _ = NewsPost.objects.filter(id=news_id).delete()
        
        if deleted:
            return Response({"message": "News deleted successfully!"})
        else:
            return Response({"error": "News not found"}, status=404)

    except Exception as e:
        print(f"Error: {e}")
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
@permission_classes([IsAdminUser])
def admin_post_news(request):
    """Admin terminal se NewsPost create karne ke liye"""
    try:
        title = request.data.get('title')
        content = request.data.get('content')
        city = request.data.get('city', 'All Pakistan')
        
        NewsPost.objects.create(
            title=title,
            content=content,
            city=city
        )
        return Response({"message": "News posted successfully!"}, status=201)
    except Exception as e:
        print(f"🚨 News Post Error: {str(e)}")
        return Response({"error": str(e)}, status=500)
   
@api_view(['GET'])
@permission_classes([IsAdminUser]) 
def get_all_users(request):
    users = User.objects.all().order_by('-date_joined') 
    data = []
    
    for u in users:
        role = "Admin" if u.is_staff else "User"
        data.append({
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": role,
            "date_joined": u.date_joined.strftime("%Y-%m-%d"), 
            "is_active": u.is_active
        })
        
    return Response(data)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_all_news(request):
    news = NewsPost.objects.all().order_by('-date_posted')
    data = []
    for n in news:
        data.append({
            "id": n.id,
            "title": n.title,
            "content": n.content,
            "city": n.city,  # 👈 Frontend ko City bhejein
            "date": n.date_posted.strftime("%Y-%m-%d")
        })
    return Response(data)

@api_view(['POST'])
@permission_classes([IsAdminUser])
def add_tip(request):
    try:
        HealthTip.objects.create(
            title=request.data.get('title'),
            description=request.data.get('description')
        )
        return Response({"message": "Tip Added!"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_all_tips(request):
    tips = HealthTip.objects.all().order_by('-date_posted')
    data = [{"id": t.id, "title": t.title, "description": t.description} for t in tips]
    return Response(data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password_api(request):
    try:
        user = request.user
        new_password = request.data.get('new_password')
        
        if not new_password or len(new_password) < 6:
            return Response({"error": "Password must be at least 6 characters"}, status=400)

        user.set_password(new_password)
        user.save()
        return Response({"message": "Password changed successfully!"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
    
@api_view(['POST'])
@permission_classes([IsAdminUser])
def update_dengue_stats(request):
    try:
        city_name = request.data.get('city_name')
        
        # Data receive karein
        active = request.data.get('active_cases')
        recovered = request.data.get('recovered')
        deaths = request.data.get('deaths')
        lat = request.data.get('latitude')
        lon = request.data.get('longitude')

        if not city_name:
            return Response({"error": "City Name Required"}, status=400)
        # Record update ya create karein
        stat, created = DengueStat.objects.get_or_create(city_name=city_name)
        
        stat.active_cases = int(active) if active else 0
        stat.recovered = int(recovered) if recovered else 0
        stat.deaths = int(deaths) if deaths else 0
        # Agar Admin ne Location bheji hai, to save karein
        if lat and lon:
            stat.latitude = float(lat)
            stat.longitude = float(lon)
            
        stat.save()

        return Response({"message": f"Updated {city_name} successfully!"})

    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
@api_view(['GET'])
@permission_classes([AllowAny])
def public_dashboard_stats(request):
    try:
        # 1. FAST CALCULATION: Database se direct sum nikalein (Loop ki zaroorat nahi)
        aggregates = DengueStat.objects.aggregate(
            total_active=Sum('active_cases'),
            total_recovered=Sum('recovered'),
            total_deaths=Sum('deaths')
        )

        # Agar DB khali ho to None ki jagah 0 use karein
        total_active = aggregates['total_active'] or 0
        total_recovered = aggregates['total_recovered'] or 0
        total_deaths = aggregates['total_deaths'] or 0

        # 2. MAP DATA PREPARATION
        # Sirf wo shehar uthayen jinki location set hai
        mapped_cities = DengueStat.objects.filter(
            latitude__isnull=False, 
            longitude__isnull=False
        )

        data = []
        for s in mapped_cities:
            data.append({
                "city_name": s.city_name,   # Frontend aksar 'city_name' expect karta hai
                "city": s.city_name,        # Backup key
                "latitude": s.latitude,     # Full spelling (Safe)
                "longitude": s.longitude,   # Full spelling (Safe)
                "lat": s.latitude,          # Short spelling (Backup)
                "lon": s.longitude,         # Short spelling (Backup)
                "active": s.active_cases,
                "recovered": s.recovered,
                "deaths": s.deaths
            })

        return Response({
            "overview": {
                "active": total_active,
                "recovered": total_recovered,
                "deaths": total_deaths
            },
            "stats": data,      # Standard naming convention
            "city_stats": data  # Aapke purane code ki compatibility ke liye
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)
@api_view(['POST'])
@permission_classes([AllowAny])
def reset_password_api(request, uidb64, token):
    try:
        new_password = request.data.get('password')
        
        # UID Decode karein aur User find karein
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
        
        # Token Verify karein
        token_generator = PasswordResetTokenGenerator()
        if not token_generator.check_token(user, token):
            return Response({"error": "Reset link is invalid or has expired."}, status=400)
            
        # Password Update karein
        user.set_password(new_password)
        user.save()
        return Response({"message": "Password successfully changed!"})
        
    except Exception as e:
        return Response({"error": "Something went wrong."}, status=400)    
    
@api_view(['POST'])
@permission_classes([AllowAny])
def send_otp_api(request):
    email = request.data.get('email')
    try:
        user = User.objects.get(email=email) # Yahan email dhoondi ja rahi hai
        
        # 6-digit Random OTP
        import random
        from django.utils import timezone
        otp_code = str(random.randint(100000, 999999))
        
        OTPRecord.objects.update_or_create(
            user=user,
            defaults={'otp': otp_code, 'created_at': timezone.now()}
        )
        
        # Email Bhejein
        from django.core.mail import send_mail
        send_mail(
            subject='DengueX - Password Reset OTP',
            message=f'Hello {user.username},\n\nYour OTP for password reset is: {otp_code}\n\nValid for 10 minutes.',
            from_email='adilmahmood7073@gmail.com', # Apna email likhna na bhoolein
            recipient_list=[email],
            fail_silently=False,
        )
        return Response({"message": "OTP sent successfully to your email!"})
        
    except User.DoesNotExist:
        return Response({"error": "User with this email does not exist."}, status=404)
    except Exception as e:
        print("❌ EMAIL SENDING ERROR:", str(e)) # Agar Gmail ka masla hua tou yahan aayega
        return Response({"error": "Failed to send email. Check Backend Terminal."}, status=500)    
  
# ==========================================
# 3. VERIFY OTP & RESET PASSWORD API
# ==========================================
@api_view(['POST'])
@permission_classes([AllowAny])
def reset_with_otp_api(request):
    email = request.data.get('email')
    otp_entered = request.data.get('otp')
    new_password = request.data.get('new_password')

    try:
        from django.contrib.auth.models import User
        user = User.objects.get(email=email)
        otp_record = OTPRecord.objects.get(user=user)

        from django.utils import timezone
        from datetime import timedelta

        # 1. Time Check karein (Kahin 10 min se purana tou nahi?)
        if timezone.now() > otp_record.created_at + timedelta(minutes=10):
            otp_record.delete()
            return Response({"error": "OTP has expired. Please request a new one."}, status=400)

        # 2. OTP Check karein
        if otp_record.otp == otp_entered:
            # Agar OTP theek hai tou Password Change karein
            user.set_password(new_password)
            user.save()
            otp_record.delete() # OTP use hone ke baad delete kar dein
            return Response({"message": "Password reset successfully!"})
        else:
            return Response({"error": "Invalid OTP code."}, status=400)

    except User.DoesNotExist:
        return Response({"error": "User not found."}, status=404)
    except OTPRecord.DoesNotExist:
        return Response({"error": "Invalid request or OTP not found."}, status=400)
    except Exception as e:
        print("❌ OTP VERIFY ERROR:", str(e))
        return Response({"error": "Something went wrong."}, status=500)  