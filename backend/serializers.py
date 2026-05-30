from rest_framework import serializers
from django.contrib.auth.models import User
from .models import MosquitoReport, NewsPost, HealthTip, DengueStat, UserProfile

# 1. News Post Serializer
class NewsPostSerializer(serializers.ModelSerializer):
    class Meta:
        model = NewsPost
        fields = '__all__'

# 2. Health Tips Serializer
class HealthTipSerializer(serializers.ModelSerializer):
    class Meta:
        model = HealthTip
        fields = '__all__'

# 3. Dengue Statistics Serializer (For Map & Analytics)
class DengueStatSerializer(serializers.ModelSerializer):
    class Meta:
        model = DengueStat
        fields = '__all__'

# 4. Main Mosquito Report Serializer (Used in Lab & Admin Console)
class MosquitoReportSerializer(serializers.ModelSerializer):
    # Username ko readable banane ke liye
    username = serializers.CharField(source='user.username', read_only=True)
    
    # Date formatting (Frontend "N/A" issue fix karne ke liye)
    formatted_date = serializers.DateTimeField(
        source='created_at', 
        format="%d %b, %I:%M %p", 
        read_only=True
    )

    class Meta:
        model = MosquitoReport
        fields = [
            'id', 
            'username', 
            'description', 
            'image', 
            'area_name', 
            'status', 
            'created_at', 
            'formatted_date',
            'latitude', 
            'longitude'
        ]
        # ID aur Timestamps system khud manage karega
        read_only_fields = ['id', 'created_at']

# 5. User Profile Serializer (Optional but useful for profile page)
class UserProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = UserProfile
        fields = '__all__'