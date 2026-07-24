import os
import re
import uuid
import random
import pathlib
import traceback
from datetime import datetime, timedelta

import torch
import torchvision.transforms as T
from PIL import Image
from fastai.learner import load_learner

# ============================================================
# DJANGO IMPORTS
# ============================================================

from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.db.models import Sum
from django.shortcuts import get_object_or_404
from django.utils.encoding import force_str
from django.utils.http import urlsafe_base64_decode

# ============================================================
# DJANGO REST FRAMEWORK
# ============================================================

from rest_framework.authtoken.models import Token
from rest_framework.decorators import (
    api_view,
    parser_classes,
    permission_classes,
)
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import (
    AllowAny,
    IsAuthenticated,
    IsAdminUser,
)
from rest_framework.response import Response

# ============================================================
# LOCAL IMPORTS
# ============================================================

from .chatbot_logic import chatbot_reply
from .serializers import MosquitoReportSerializer
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


# ============================================================
# MOSQUITO CLASSIFICATION MODEL
# ============================================================

# settings.BASE_DIR points to:
# project/backend/
#
# The trained model is located at:
# project/DengueX-Modules/mosquito_detection/
# Model_Training/Efficient_net_b0/mosquito_classifier.pkl

MODEL_DIR = os.path.join(
    settings.BASE_DIR.parent,
    "DengueX-Modules",
    "mosquito_detection",
    "Model_Training",
    "Efficient_net_b0",
)

FASTAI_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mosquito_classifier.pkl",
)

model = None
labels = []
fastai_preprocess = None


def load_local_model():
    """
    Lazily load the FastAI mosquito classification model.

    The model is loaded only when the detection API is first called.
    This prevents Django startup from unnecessarily loading the ML model.
    """

    global model, labels, fastai_preprocess

    try:
        if not os.path.exists(FASTAI_MODEL_PATH):
            print(
                f"❌ Mosquito model not found: "
                f"{FASTAI_MODEL_PATH}"
            )
            return

        # FastAI models exported on Linux may contain PosixPath
        # objects. Convert them only when running on Windows.
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath

        learner = load_learner(
            FASTAI_MODEL_PATH,
            cpu=True,
        )

        model = learner.model.eval()

        labels = [
            str(label)
            for label in getattr(
                learner.dls,
                "vocab",
                [],
            )
        ]

        if not labels:
            print(
                "❌ Mosquito model loaded, "
                "but no class labels were found."
            )

            model = None
            return

        fastai_preprocess = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

        print(
            "✅ FastAI mosquito model loaded successfully"
        )

        print(
            f"✅ Model path: {FASTAI_MODEL_PATH}"
        )

        print(
            f"✅ Classes: {labels}"
        )

    except Exception as exc:
        model = None
        labels = []
        fastai_preprocess = None

        print(
            f"❌ FastAI model loading failed: {exc}"
        )

        traceback.print_exc()


# ============================================================
# MOSQUITO DETECTION API
# ============================================================


@api_view(["POST"])
@parser_classes(
    [
        MultiPartParser,
        FormParser,
    ]
)
@permission_classes([AllowAny])
def detect_mosquito_specie(request):

    global model, labels, fastai_preprocess

    try:
        image_file = request.FILES.get("image")

        if not image_file:
            return Response(
                {
                    "result": "ERROR",
                    "message": "Specimen missing.",
                },
                status=400,
            )

        # Lazy model loading
        if model is None:
            load_local_model()

        if (
            model is None
            or fastai_preprocess is None
            or not labels
        ):
            return Response(
                {
                    "result": "ERROR",
                    "message": (
                        "Mosquito classification "
                        "model is not available."
                    ),
                },
                status=503,
            )

        raw_img = Image.open(
            image_file
        ).convert("RGB")

        input_tensor = (
            fastai_preprocess(raw_img)
            .unsqueeze(0)
        )

        with torch.no_grad():
            logits = model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

        class_idx = int(
            torch.argmax(
                probabilities
            ).item()
        )

        if class_idx >= len(labels):
            raise RuntimeError(
                "Prediction index does not "
                "match model labels."
            )

        identified_specie = labels[class_idx]

        confidence = (
            float(
                probabilities[
                    class_idx
                ].item()
            )
            * 100
        )

        normalized_specie = (
            identified_specie
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        is_dengue = (
            normalized_specie == "aedes"
        )

        risk_map = {
            "aedes":
                "HIGH (Dengue Vector)",

            "culex":
                "MODERATE "
                "(West Nile/Filariasis)",

            "anopheles":
                "HIGH (Malaria Vector)",

            "non_aedes":
                "LOW "
                "(Not classified as "
                "Aedes dengue vector)",

            "unknown":
                "LOW "
                "(Unable to classify "
                "confidently)",
        }

        habitat_map = {
            "aedes":
                "Stagnant water in "
                "containers, pots, and tires.",

            "culex":
                "Polluted water, drains, "
                "and ditches.",

            "anopheles":
                "Clean, slow-moving "
                "fresh water.",

            "non_aedes":
                "General outdoor breeding "
                "areas; remove standing water.",

            "unknown":
                "Image unclear. Capture "
                "a closer and well-lit "
                "specimen image.",
        }

        return Response(
            {
                "result":
                    identified_specie.upper(),

                "common_name":
                    f"{identified_specie} Mosquito",

                "specie":
                    identified_specie,

                "is_dengue":
                    is_dengue,

                "risk":
                    risk_map.get(
                        normalized_specie,
                        "LOW",
                    ),

                "habitat":
                    habitat_map.get(
                        normalized_specie,
                        "Natural surroundings",
                    ),

                "message":
                    (
                        f"Identified "
                        f"{identified_specie} "
                        f"with "
                        f"{confidence:.1f}% "
                        f"confidence."
                    ),

                "confidence":
                    f"{confidence:.1f}%",
            },
            status=200,
        )

    except Exception as exc:

        print(
            f"🚨 Prediction Error: {exc}"
        )

        traceback.print_exc()

        return Response(
            {
                "result": "ERROR",
                "message":
                    "Unable to process "
                    "the specimen.",
            },
            status=500,
        )


# ============================================================
# CHATBOT API
# ============================================================


@api_view(["POST"])
@permission_classes([AllowAny])
def chatbot_response(request):

    user_message = (
        request.data
        .get("message", "")
        .strip()
    )

    session_id = request.data.get(
        "session_id"
    )

    if not user_message:
        return Response(
            {
                "error":
                    "Message is required"
            },
            status=400,
        )

    try:
        bot_reply = chatbot_reply(
            user_message
        )

        intent = "non-urgent"
        confidence = 0.0

    except Exception as exc:

        print(
            f"❌ Chatbot Error: {exc}"
        )

        bot_reply = (
            "I'm having trouble "
            "processing that locally. "
            "Please try again."
        )

        intent = "error"
        confidence = 0.0

    current_session_id = session_id

    if request.user.is_authenticated:

        session = None

        if session_id:

            session = (
                ChatSession.objects
                .filter(
                    id=session_id,
                    user=request.user,
                )
                .first()
            )

        if not session:

            session = (
                ChatSession.objects.create(
                    user=request.user,
                    title=user_message[:30],
                )
            )

            current_session_id = (
                session.id
            )

        ChatMessage.objects.create(
            session=session,
            role="user",
            content=user_message,
        )

        ChatMessage.objects.create(
            session=session,
            role="assistant",
            content=bot_reply,
        )

    return Response(
        {
            "response":
                bot_reply,

            "intent":
                intent,

            "confidence":
                confidence,

            "session_id":
                current_session_id,
        }
    )


# ============================================================
# CHAT SESSION MANAGEMENT
# ============================================================


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_chat_sessions(request):

    sessions = (
        ChatSession.objects
        .filter(user=request.user)
        .order_by("-created_at")
    )

    return Response(
        [
            {
                "id": session.id,
                "title": session.title,
            }
            for session in sessions
        ]
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_chat_messages(
    request,
    session_id,
):

    try:

        session = (
            ChatSession.objects.get(
                id=session_id,
                user=request.user,
            )
        )

        messages = (
            session.messages
            .all()
            .order_by("timestamp")
        )

        return Response(
            [
                {
                    "sender":
                        (
                            "bot"
                            if message.role
                            == "assistant"
                            else "user"
                        ),

                    "text":
                        message.content,
                }
                for message in messages
            ]
        )

    except ChatSession.DoesNotExist:

        return Response(
            {
                "error":
                    "Chat not found"
            },
            status=404,
        )


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def delete_chat_session(
    request,
    session_id,
):

    ChatSession.objects.filter(
        id=session_id,
        user=request.user,
    ).delete()

    return Response(
        {
            "message": "Deleted"
        },
        status=200,
    )


# ============================================================
# MOSQUITO REPORT APIs
# ============================================================


@api_view(["GET"])
def get_report_details(
    request,
    pk,
):

    try:

        report = get_object_or_404(
            MosquitoReport,
            pk=pk,
        )

        serializer = (
            MosquitoReportSerializer(
                report
            )
        )

        return Response(
            serializer.data
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def submit_report_api(request):

    try:

        description = (
            request.data.get(
                "description",
                "",
            )
        )

        area = (
            request.data.get(
                "area_name",
                "Unknown",
            )
        )

        latitude = (
            request.data.get(
                "latitude"
            )
        )

        longitude = (
            request.data.get(
                "longitude"
            )
        )

        image = (
            request.FILES.get(
                "image"
            )
        )

        report = (
            MosquitoReport.objects.create(
                user=request.user,

                image=image,

                description=description,

                area_name=area,

                latitude=(
                    float(latitude)
                    if latitude
                    else None
                ),

                longitude=(
                    float(longitude)
                    if longitude
                    else None
                ),
            )
        )

        return Response(
            {
                "message":
                    "Log Created",

                "id":
                    report.id,
            },
            status=201,
        )

    except Exception as exc:

        print(
            f"🚨 Submit Error: {exc}"
        )

        return Response(
            {
                "error": str(exc)
            },
            status=400,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_reports_api(request):

    reports = (
        MosquitoReport.objects
        .filter(user=request.user)
        .order_by("-created_at")
    )

    data = [
        {
            "id":
                report.id,

            "area_name":
                report.area_name,

            "status":
                report.status,

            "image":
                (
                    report.image.url
                    if report.image
                    else None
                ),

            "description":
                report.description,

            "latitude":
                report.latitude,

            "longitude":
                report.longitude,

            "created_at":
                report.created_at,
        }
        for report in reports
    ]

    return Response(data)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def delete_report(
    request,
    pk,
):

    try:

        report = (
            MosquitoReport.objects
            .filter(pk=pk)
            .first()
        )

        if not report:

            return Response(
                {
                    "error":
                        "Report not found"
                },
                status=404,
            )

        report.delete()

        return Response(
            {
                "message":
                    "Successfully erased "
                    "from core."
            },
            status=200,
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def delete_all_reports(request):

    MosquitoReport.objects.filter(
        user=request.user
    ).delete()

    return Response(
        {
            "message":
                "Core Formatted"
        },
        status=200,
    )


# ============================================================
# ADMIN REPORT APIs
# ============================================================


@api_view(["GET"])
@permission_classes([IsAdminUser])
def admin_get_all_reports(request):

    reports = (
        MosquitoReport.objects
        .all()
        .order_by("-created_at")
    )

    serializer = (
        MosquitoReportSerializer(
            reports,
            many=True,
        )
    )

    return Response(
        {
            "total":
                reports.count(),

            "pending":
                reports.filter(
                    status="pending"
                ).count(),

            "resolved":
                reports.filter(
                    status="resolved"
                ).count(),

            "rejected":
                reports.filter(
                    status="rejected"
                ).count(),

            "reports":
                serializer.data,
        }
    )


@api_view(["POST"])
@permission_classes([IsAdminUser])
def admin_update_report_status(
    request,
):

    try:

        report_id = (
            request.data.get("id")
        )

        new_status = (
            request.data
            .get("status", "")
            .lower()
        )

        if not report_id or not new_status:

            return Response(
                {
                    "error":
                        "Report ID and status "
                        "are required."
                },
                status=400,
            )

        report = (
            MosquitoReport.objects
            .get(id=report_id)
        )

        report.status = new_status
        report.save()

        return Response(
            {
                "message":
                    f"Status updated "
                    f"to {new_status}"
            },
            status=200,
        )

    except MosquitoReport.DoesNotExist:

        return Response(
            {
                "error":
                    "Report not found"
            },
            status=404,
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


# ============================================================
# PROFILE IMAGE
# ============================================================


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_profile_image(request):

    try:

        profile, _ = (
            UserProfile.objects
            .get_or_create(
                user=request.user
            )
        )

        if (
            "profile_image"
            not in request.FILES
        ):

            return Response(
                {
                    "error":
                        "No image found "
                        "in request"
                },
                status=400,
            )

        profile.profile_image = (
            request.FILES[
                "profile_image"
            ]
        )

        profile.save()

        image_url = (
            request
            .build_absolute_uri(
                profile
                .profile_image
                .url
            )
        )

        return Response(
            {
                "profile_image":
                    image_url,

                "message":
                    "Success! Image updated.",
            },
            status=200,
        )

    except Exception as exc:

        print(
            f"Profile image error: {exc}"
        )

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


# ============================================================
# AUTHENTICATION
# ============================================================


@api_view(["POST"])
@permission_classes([AllowAny])
def login_api(request):

    username = (
        request.data.get("username")
    )

    password = (
        request.data.get("password")
    )

    user = authenticate(
        username=username,
        password=password,
    )

    if user is None:

        return Response(
            {
                "error":
                    "Invalid username "
                    "or password"
            },
            status=401,
        )

    token, _ = (
        Token.objects
        .get_or_create(user=user)
    )

    return Response(
        {
            "message":
                "Login successful",

            "username":
                user.username,

            "token":
                token.key,

            "is_admin":
                user.is_staff,
        }
    )


@api_view(["POST"])
@permission_classes([AllowAny])
def signup_api(request):

    username = (
        request.data.get("username")
    )

    password = (
        request.data.get("password")
    )

    email = (
        request.data.get("email")
    )

    if (
        not username
        or not password
        or not email
    ):

        return Response(
            {
                "error":
                    "Username, Password "
                    "and Email are required."
            },
            status=400,
        )

    if (
        User.objects
        .filter(username=username)
        .exists()
    ):

        return Response(
            {
                "error":
                    "Username is already taken."
            },
            status=400,
        )

    try:

        user = (
            User.objects.create_user(
                username=username,
                password=password,
                email=email,
            )
        )

        UserProfile.objects.create(
            user=user,
            email=email,
        )

        token, _ = (
            Token.objects
            .get_or_create(user=user)
        )

        return Response(
            {
                "message":
                    "Account created successfully!",

                "token":
                    token.key,

                "username":
                    user.username,
            },
            status=201,
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


@api_view(["POST"])
@permission_classes([AllowAny])
def google_login_api(request):

    email = (
        request.data.get("email")
    )

    name = (
        request.data.get("name")
    )

    if not email:

        return Response(
            {
                "error":
                    "Email is required"
            },
            status=400,
        )

    user = (
        User.objects
        .filter(username=email)
        .first()
    )

    if user:

        token, _ = (
            Token.objects
            .get_or_create(user=user)
        )

        return Response(
            {
                "message":
                    "Login Successful",

                "username":
                    user.username,

                "token":
                    token.key,
            }
        )

    try:

        random_password = (
            str(uuid.uuid4())
        )

        user = (
            User.objects.create_user(
                username=email,
                email=email,
                password=random_password,
            )
        )

        UserProfile.objects.create(
            user=user,
            full_name=name,
        )

        token, _ = (
            Token.objects
            .get_or_create(user=user)
        )

        return Response(
            {
                "message":
                    "Account Created",

                "username":
                    user.username,

                "token":
                    token.key,
            }
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


# ============================================================
# ADMIN PASSWORD CHANGE
# ============================================================


@api_view(["POST"])
@permission_classes([IsAdminUser])
def admin_change_password_api(request):

    try:

        user_id = (
            request.data.get("user_id")
        )

        new_password = (
            request.data.get(
                "new_password"
            )
        )

        if (
            not user_id
            or not new_password
        ):

            return Response(
                {
                    "error":
                        "User ID and New Password "
                        "are required."
                },
                status=400,
            )

        if len(new_password) < 6:

            return Response(
                {
                    "error":
                        "Password must be at least "
                        "6 characters long."
                },
                status=400,
            )

        try:

            user = User.objects.get(
                id=user_id
            )

        except User.DoesNotExist:

            return Response(
                {
                    "error":
                        "User not found."
                },
                status=404,
            )

        user.set_password(
            new_password
        )

        user.save()

        return Response(
            {
                "message":
                    (
                        f"Password for "
                        f"{user.username} "
                        f"updated successfully!"
                    )
            },
            status=200,
        )

    except Exception as exc:

        print(
            "Admin Password "
            f"Change Error: {exc}"
        )

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


# ============================================================
# PROFILE
# ============================================================


@api_view(["POST", "GET"])
@permission_classes([IsAuthenticated])
def get_profile(request):

    try:

        user = request.user

        profile, _ = (
            UserProfile.objects
            .get_or_create(user=user)
        )

        data = {
            "username":
                user.username,

            "email":
                user.email,

            "full_name":
                profile.full_name,

            "age":
                profile.age,

            "blood_group":
                profile.blood_group,

            "city":
                profile.city,

            "emergency_contact":
                profile.emergency_contact,

            "previous_infection":
                profile.previous_infection,

            "comorbidities":
                profile.comorbidities,

            "travel_history":
                profile.travel_history,
        }

        return Response(data)

    except Exception as exc:

        print(
            f"Get Profile Error: {exc}"
        )

        return Response(
            {
                "error": str(exc)
            },
            status=400,
        )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_profile_api(request):

    try:

        profile, _ = (
            UserProfile.objects
            .get_or_create(
                user=request.user
            )
        )

        profile.full_name = (
            request.data.get(
                "full_name",
                profile.full_name,
            )
        )

        age = (
            request.data.get("age")
        )

        if age not in (
            None,
            "",
        ):
            profile.age = int(age)

        profile.blood_group = (
            request.data.get(
                "blood_group",
                profile.blood_group,
            )
        )

        if hasattr(
            profile,
            "is_vaccinated",
        ):
            profile.is_vaccinated = (
                request.data.get(
                    "is_vaccinated",
                    profile.is_vaccinated,
                )
            )

        if hasattr(
            profile,
            "recent_test_date",
        ):
            profile.recent_test_date = (
                request.data.get(
                    "recent_test_date",
                    profile.recent_test_date,
                )
            )

        profile.save()

        return Response(
            {
                "message":
                    "Profile updated successfully!"
            }
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=400,
        )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_profile(request):

    try:

        user = request.user

        profile, _ = (
            UserProfile.objects
            .get_or_create(user=user)
        )

        data = request.data

        email = (
            data.get("email")
        )

        if (
            email
            and email.strip()
        ):
            user.email = email.strip()
            user.save()

        age_value = (
            data.get("age")
        )

        if age_value == "":
            profile.age = None

        elif age_value is not None:
            profile.age = int(age_value)

        profile.full_name = (
            data.get(
                "full_name",
                profile.full_name,
            )
        )

        profile.blood_group = (
            data.get(
                "blood_group",
                profile.blood_group,
            )
        )

        profile.city = (
            data.get(
                "city",
                profile.city,
            )
        )

        profile.emergency_contact = (
            data.get(
                "emergency_contact",
                profile.emergency_contact,
            )
        )

        profile.previous_infection = (
            data.get(
                "previous_infection",
                profile.previous_infection,
            )
        )

        profile.comorbidities = (
            data.get(
                "comorbidities",
                profile.comorbidities,
            )
        )

        profile.travel_history = (
            data.get(
                "travel_history",
                profile.travel_history,
            )
        )

        profile.save()

        return Response(
            {
                "message":
                    "Profile updated successfully!"
            }
        )

    except Exception as exc:

        print(
            f"❌ PROFILE UPDATE ERROR: {exc}"
        )

        return Response(
            {
                "error": str(exc)
            },
            status=400,
        )


# ============================================================
# DASHBOARD
# ============================================================


@api_view(["GET"])
@permission_classes([AllowAny])
def dashboard_stats_api(request):

    try:

        aggregates = (
            DengueStat.objects.aggregate(
                active=Sum(
                    "active_cases"
                ),
                recovered=Sum(
                    "recovered"
                ),
                deaths=Sum(
                    "deaths"
                ),
            )
        )

        active = (
            aggregates["active"]
            or 0
        )

        recovered = (
            aggregates["recovered"]
            or 0
        )

        deaths = (
            aggregates["deaths"]
            or 0
        )

        total_cases = (
            active
            + recovered
            + deaths
        )

        city_stats = (
            DengueStat.objects.all()
        )

        city_data = [
            {
                "id":
                    stat.id,

                "city":
                    stat.city_name,

                "active":
                    stat.active_cases,

                "recovered":
                    stat.recovered,

                "deaths":
                    stat.deaths,

                "latitude":
                    stat.latitude,

                "longitude":
                    stat.longitude,
            }
            for stat in city_stats
        ]

        tips = (
            HealthTip.objects
            .all()
            .order_by("-date_posted")[:5]
        )

        tips_data = [
            {
                "id":
                    tip.id,

                "title":
                    tip.title,

                "description":
                    tip.description,

                "date":
                    (
                        tip.date_posted
                        .strftime("%Y-%m-%d")
                        if tip.date_posted
                        else ""
                    ),
            }
            for tip in tips
        ]

        return Response(
            {
                "summary": {
                    "total_reports":
                        total_cases,

                    "active":
                        active,

                    "recovered":
                        recovered,

                    "deaths":
                        deaths,
                },

                "stats":
                    city_data,

                "city_stats":
                    city_data,

                "health_tips":
                    tips_data,
            }
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


# ============================================================
# ANALYTICS
# ============================================================


@api_view(["GET"])
@permission_classes([AllowAny])
def analytics_data_api(request):

    try:

        total_active_admin = (
            DengueStat.objects
            .aggregate(
                total=Sum(
                    "active_cases"
                )
            )["total"]
            or 0
        )

        chart_data = []

        end_date = datetime.now()

        start_date = (
            end_date
            - timedelta(days=29)
        )

        total_days = (
            end_date
            - start_date
        ).days + 1

        for index in range(
            total_days
        ):

            date_obj = (
                start_date
                + timedelta(
                    days=index
                )
            )

            display_date = (
                date_obj.strftime(
                    "%b %d"
                )
            )

            days_from_today = (
                end_date.date()
                - date_obj.date()
            ).days

            if days_from_today == 0:

                estimated = (
                    total_active_admin
                )

            else:

                decrease = int(
                    total_active_admin
                    * 0.05
                    * days_from_today
                )

                estimated = max(
                    0,
                    total_active_admin
                    - decrease,
                )

            chart_data.append(
                {
                    "date":
                        display_date,

                    "cases":
                        estimated,
                }
            )

        return Response(
            chart_data
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


# ============================================================
# PUBLIC DASHBOARD
# ============================================================


@api_view(["GET"])
@permission_classes([AllowAny])
def public_dashboard_stats(request):

    try:

        aggregates = (
            DengueStat.objects.aggregate(
                total_active=Sum(
                    "active_cases"
                ),

                total_recovered=Sum(
                    "recovered"
                ),

                total_deaths=Sum(
                    "deaths"
                ),
            )
        )

        total_active = (
            aggregates[
                "total_active"
            ]
            or 0
        )

        total_recovered = (
            aggregates[
                "total_recovered"
            ]
            or 0
        )

        total_deaths = (
            aggregates[
                "total_deaths"
            ]
            or 0
        )

        mapped_cities = (
            DengueStat.objects
            .filter(
                latitude__isnull=False,
                longitude__isnull=False,
            )
        )

        data = []

        for stat in mapped_cities:

            data.append(
                {
                    "city_name":
                        stat.city_name,

                    "city":
                        stat.city_name,

                    "latitude":
                        stat.latitude,

                    "longitude":
                        stat.longitude,

                    "lat":
                        stat.latitude,

                    "lon":
                        stat.longitude,

                    "active":
                        stat.active_cases,

                    "recovered":
                        stat.recovered,

                    "deaths":
                        stat.deaths,
                }
            )

        return Response(
            {
                "overview": {
                    "active":
                        total_active,

                    "recovered":
                        total_recovered,

                    "deaths":
                        total_deaths,
                },

                "stats":
                    data,

                "city_stats":
                    data,
            }
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


# ============================================================
# HEALTH TIPS
# ============================================================


@api_view(["POST"])
@permission_classes([IsAdminUser])
def admin_delete_tip(request):

    try:

        tip_id = (
            request.data.get("id")
        )

        if not tip_id:

            return Response(
                {
                    "error":
                        "Tip ID is required"
                },
                status=400,
            )

        deleted, _ = (
            HealthTip.objects
            .filter(id=tip_id)
            .delete()
        )

        if deleted:

            return Response(
                {
                    "message":
                        "Tip deleted successfully!"
                }
            )

        return Response(
            {
                "error":
                    "Tip not found"
            },
            status=404,
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


@api_view(["POST"])
@permission_classes([IsAdminUser])
def add_tip(request):

    try:

        HealthTip.objects.create(
            title=request.data.get(
                "title"
            ),

            description=request.data.get(
                "description"
            ),
        )

        return Response(
            {
                "message":
                    "Tip Added!"
            }
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def get_all_tips(request):

    tips = (
        HealthTip.objects
        .all()
        .order_by("-date_posted")
    )

    data = [
        {
            "id":
                tip.id,

            "title":
                tip.title,

            "description":
                tip.description,
        }
        for tip in tips
    ]

    return Response(data)


# ============================================================
# NEWS
# ============================================================


@api_view(["GET"])
@permission_classes([AllowAny])
def get_news_api(request):

    news_list = (
        NewsPost.objects
        .all()
        .order_by("-date_posted")
    )

    data = [
        {
            "id":
                news.id,

            "title":
                news.title,

            "city":
                news.city,

            "content":
                news.content,

            "date":
                (
                    news.date_posted
                    .strftime(
                        "%d %b, %Y"
                    )
                    if news.date_posted
                    else "N/A"
                ),
        }
        for news in news_list
    ]

    return Response(data)


@api_view(["POST"])
@permission_classes([IsAdminUser])
def admin_delete_news(request):

    try:

        news_id = (
            request.data.get("id")
        )

        if not news_id:

            return Response(
                {
                    "error":
                        "News ID is required"
                },
                status=400,
            )

        deleted, _ = (
            NewsPost.objects
            .filter(id=news_id)
            .delete()
        )

        if deleted:

            return Response(
                {
                    "message":
                        "News deleted successfully!"
                }
            )

        return Response(
            {
                "error":
                    "News not found"
            },
            status=404,
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


@api_view(["POST"])
@permission_classes([IsAdminUser])
def admin_post_news(request):

    try:

        title = (
            request.data.get("title")
        )

        content = (
            request.data.get("content")
        )

        city = (
            request.data.get(
                "city",
                "All Pakistan",
            )
        )

        NewsPost.objects.create(
            title=title,
            content=content,
            city=city,
        )

        return Response(
            {
                "message":
                    "News posted successfully!"
            },
            status=201,
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_all_news(request):

    news = (
        NewsPost.objects
        .all()
        .order_by("-date_posted")
    )

    data = [
        {
            "id":
                item.id,

            "title":
                item.title,

            "content":
                item.content,

            "city":
                item.city,

            "date":
                item.date_posted
                .strftime("%Y-%m-%d"),
        }
        for item in news
    ]

    return Response(data)


# ============================================================
# ADMIN USERS
# ============================================================


@api_view(["GET"])
@permission_classes([IsAdminUser])
def get_all_users(request):

    users = (
        User.objects
        .all()
        .order_by("-date_joined")
    )

    data = []

    for user in users:

        data.append(
            {
                "id":
                    user.id,

                "username":
                    user.username,

                "email":
                    user.email,

                "role":
                    (
                        "Admin"
                        if user.is_staff
                        else "User"
                    ),

                "date_joined":
                    user.date_joined
                    .strftime(
                        "%Y-%m-%d"
                    ),

                "is_active":
                    user.is_active,
            }
        )

    return Response(data)


@api_view(["POST"])
@permission_classes([IsAdminUser])
def toggle_user_block_status(request):

    try:

        user_id = (
            request.data.get(
                "user_id"
            )
        )

        user = (
            User.objects.get(
                id=user_id
            )
        )

        if user.is_superuser:

            return Response(
                {
                    "error":
                        "Cannot block "
                        "Super Admin"
                },
                status=400,
            )

        user.is_active = (
            not user.is_active
        )

        user.save()

        status_message = (
            "Unblocked"
            if user.is_active
            else "Blocked"
        )

        return Response(
            {
                "message":
                    (
                        f"User "
                        f"{status_message} "
                        f"successfully"
                    ),

                "is_active":
                    user.is_active,
            }
        )

    except User.DoesNotExist:

        return Response(
            {
                "error":
                    "User not found"
            },
            status=404,
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


# ============================================================
# DENGUE STATISTICS ADMIN
# ============================================================


@api_view(["POST"])
@permission_classes([IsAdminUser])
def admin_update_stats(request):

    try:

        raw_city = (
            request.data.get(
                "city_name"
            )
        )

        if not raw_city:

            return Response(
                {
                    "error":
                        "City Name is required"
                },
                status=400,
            )

        city_name = (
            raw_city
            .strip()
            .title()
        )

        active = (
            request.data.get(
                "active_cases"
            )
        )

        recovered = (
            request.data.get(
                "recovered"
            )
        )

        deaths = (
            request.data.get(
                "deaths"
            )
        )

        latitude = (
            request.data.get(
                "latitude"
            )
        )

        longitude = (
            request.data.get(
                "longitude"
            )
        )

        defaults = {
            "active_cases":
                (
                    int(active)
                    if active
                    else 0
                ),

            "recovered":
                (
                    int(recovered)
                    if recovered
                    else 0
                ),

            "deaths":
                (
                    int(deaths)
                    if deaths
                    else 0
                ),
        }

        if latitude and longitude:

            defaults["latitude"] = (
                float(latitude)
            )

            defaults["longitude"] = (
                float(longitude)
            )

        obj, created = (
            DengueStat.objects
            .update_or_create(
                city_name=city_name,
                defaults=defaults,
            )
        )

        action = (
            "Created"
            if created
            else "Updated"
        )

        return Response(
            {
                "message":
                    (
                        f"{action} "
                        f"{city_name} "
                        f"successfully!"
                    )
            }
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


@api_view(["POST"])
@permission_classes([IsAdminUser])
def admin_delete_city(request):

    try:

        city_id = (
            request.data.get("id")
        )

        if not city_id:

            return Response(
                {
                    "error":
                        "ID required"
                },
                status=400,
            )

        DengueStat.objects.filter(
            id=city_id
        ).delete()

        return Response(
            {
                "message":
                    "City deleted successfully!"
            }
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


@api_view(["POST"])
@permission_classes([IsAdminUser])
def update_dengue_stats(request):

    try:

        city_name = (
            request.data.get(
                "city_name"
            )
        )

        if not city_name:

            return Response(
                {
                    "error":
                        "City Name Required"
                },
                status=400,
            )

        active = (
            request.data.get(
                "active_cases"
            )
        )

        recovered = (
            request.data.get(
                "recovered"
            )
        )

        deaths = (
            request.data.get(
                "deaths"
            )
        )

        latitude = (
            request.data.get(
                "latitude"
            )
        )

        longitude = (
            request.data.get(
                "longitude"
            )
        )

        stat, _ = (
            DengueStat.objects
            .get_or_create(
                city_name=city_name
            )
        )

        stat.active_cases = (
            int(active)
            if active
            else 0
        )

        stat.recovered = (
            int(recovered)
            if recovered
            else 0
        )

        stat.deaths = (
            int(deaths)
            if deaths
            else 0
        )

        if latitude and longitude:

            stat.latitude = (
                float(latitude)
            )

            stat.longitude = (
                float(longitude)
            )

        stat.save()

        return Response(
            {
                "message":
                    (
                        f"Updated "
                        f"{city_name} "
                        f"successfully!"
                    )
            }
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


# ============================================================
# USER PASSWORD CHANGE
# ============================================================


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def change_password_api(request):

    try:

        user = request.user

        new_password = (
            request.data.get(
                "new_password"
            )
        )

        if (
            not new_password
            or len(new_password) < 6
        ):

            return Response(
                {
                    "error":
                        "Password must be at least "
                        "6 characters"
                },
                status=400,
            )

        user.set_password(
            new_password
        )

        user.save()

        return Response(
            {
                "message":
                    "Password changed successfully!"
            }
        )

    except Exception as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=500,
        )


# ============================================================
# PASSWORD RESET USING DJANGO TOKEN
# ============================================================


@api_view(["POST"])
@permission_classes([AllowAny])
def reset_password_api(
    request,
    uidb64,
    token,
):

    try:

        new_password = (
            request.data.get(
                "password"
            )
        )

        if (
            not new_password
            or len(new_password) < 6
        ):

            return Response(
                {
                    "error":
                        "Password must be at least "
                        "6 characters."
                },
                status=400,
            )

        uid = force_str(
            urlsafe_base64_decode(
                uidb64
            )
        )

        user = User.objects.get(
            pk=uid
        )

        token_generator = (
            PasswordResetTokenGenerator()
        )

        if not token_generator.check_token(
            user,
            token,
        ):

            return Response(
                {
                    "error":
                        "Reset link is invalid "
                        "or has expired."
                },
                status=400,
            )

        user.set_password(
            new_password
        )

        user.save()

        return Response(
            {
                "message":
                    "Password successfully changed!"
            }
        )

    except Exception:

        return Response(
            {
                "error":
                    "Something went wrong."
            },
            status=400,
        )


# ============================================================
# OTP PASSWORD RESET
# ============================================================


@api_view(["POST"])
@permission_classes([AllowAny])
def send_otp_api(request):

    email = (
        request.data.get("email")
    )

    if not email:

        return Response(
            {
                "error":
                    "Email is required."
            },
            status=400,
        )

    try:

        user = User.objects.get(
            email=email
        )

        from django.utils import timezone
        from django.core.mail import send_mail

        otp_code = str(
            random.randint(
                100000,
                999999,
            )
        )

        OTPRecord.objects.update_or_create(
            user=user,

            defaults={
                "otp":
                    otp_code,

                "created_at":
                    timezone.now(),
            },
        )

        send_mail(
            subject=(
                "DengueX - "
                "Password Reset OTP"
            ),

            message=(
                f"Hello {user.username},\n\n"
                f"Your OTP for password reset is: "
                f"{otp_code}\n\n"
                f"Valid for 10 minutes."
            ),

            from_email=(
                settings.DEFAULT_FROM_EMAIL
            ),

            recipient_list=[
                email
            ],

            fail_silently=False,
        )

        return Response(
            {
                "message":
                    "OTP sent successfully "
                    "to your email!"
            }
        )

    except User.DoesNotExist:

        return Response(
            {
                "error":
                    "User with this email "
                    "does not exist."
            },
            status=404,
        )

    except Exception as exc:

        print(
            f"❌ EMAIL SENDING ERROR: {exc}"
        )

        return Response(
            {
                "error":
                    "Failed to send email."
            },
            status=500,
        )


@api_view(["POST"])
@permission_classes([AllowAny])
def reset_with_otp_api(request):

    email = (
        request.data.get("email")
    )

    otp_entered = (
        request.data.get("otp")
    )

    new_password = (
        request.data.get(
            "new_password"
        )
    )

    if (
        not email
        or not otp_entered
        or not new_password
    ):

        return Response(
            {
                "error":
                    "Email, OTP and new password "
                    "are required."
            },
            status=400,
        )

    if len(new_password) < 6:

        return Response(
            {
                "error":
                    "Password must be at least "
                    "6 characters."
            },
            status=400,
        )

    try:

        from django.utils import timezone

        user = User.objects.get(
            email=email
        )

        otp_record = (
            OTPRecord.objects.get(
                user=user
            )
        )

        if (
            timezone.now()
            >
            otp_record.created_at
            + timedelta(minutes=10)
        ):

            otp_record.delete()

            return Response(
                {
                    "error":
                        "OTP has expired. "
                        "Please request a new one."
                },
                status=400,
            )

        if (
            otp_record.otp
            != otp_entered
        ):

            return Response(
                {
                    "error":
                        "Invalid OTP code."
                },
                status=400,
            )

        user.set_password(
            new_password
        )

        user.save()

        otp_record.delete()

        return Response(
            {
                "message":
                    "Password reset successfully!"
            }
        )

    except User.DoesNotExist:

        return Response(
            {
                "error":
                    "User not found."
            },
            status=404,
        )

    except OTPRecord.DoesNotExist:

        return Response(
            {
                "error":
                    "Invalid request "
                    "or OTP not found."
            },
            status=400,
        )

    except Exception as exc:

        print(
            f"❌ OTP VERIFY ERROR: {exc}"
        )

        return Response(
            {
                "error":
                    "Something went wrong."
            },
            status=500,
        )