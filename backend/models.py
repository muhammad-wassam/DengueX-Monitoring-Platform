from django.db import models
from django.contrib.auth.models import User


# models.py mein
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    email = models.EmailField(blank=True, null=True) 
    full_name = models.CharField(max_length=100, blank=True, null=True) # 👈 null=True add karein
    age = models.IntegerField(null=True, blank=True)
    blood_group = models.CharField(max_length=10, blank=True, null=True) # 👈 null=True add karein
    city = models.CharField(max_length=100, blank=True, null=True) # 👈 null=True add karein
    emergency_contact = models.CharField(max_length=20, blank=True, null=True) # 👈 null=True add karein
    previous_infection = models.CharField(max_length=20, blank=True, null=True) # 👈 null=True add karein
    comorbidities = models.TextField(blank=True, null=True) # 👈 null=True add karein
    travel_history = models.CharField(max_length=20, blank=True, null=True) # 👈 null=True add karein
    profile_image = models.ImageField(upload_to='profile_pics/', null=True, blank=True)
    
    def __str__(self):
        return self.user.username

class DengueStat(models.Model):
    city_name = models.CharField(max_length=100, unique=True)
    active_cases = models.IntegerField(default=0)
    recovered = models.IntegerField(default=0)
    deaths = models.IntegerField(default=0)
    
    # ✅ Location Fields (Map ke liye zaroori)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    
    # ✅ Last Updated Time (Chart ke liye zaroori)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.city_name

# ✅ Chat Session (Jo Sidebar mein dikhega: "Dengue Symptoms", "Prevention", etc.)
class MosquitoReport(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('resolved', 'Resolved'),
        ('rejected', 'Rejected'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='reports/')
    description = models.TextField(null=True, blank=True)
    area_name = models.CharField(max_length=255)
    is_deleted_by_user = models.BooleanField(default=False)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    def __str__(self):
        return f"{self.area_name} - {self.status}"

class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, blank=True) # Chat ka Topic
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.title}"

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, related_name='messages', on_delete=models.CASCADE)
    role = models.CharField(max_length=50) # 'user' or 'assistant'
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.role}: {self.content[:20]}..." 
   
# Create your models here.
class NewsPost(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    city = models.CharField(max_length=100, default="All Pakistan")  # 👈 New Field
    date_posted = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.title} ({self.city})"
# ✅ ADD THIS TOO (IF YOU WANT HEALTH TIPS SEPARATE)
class HealthTip(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    date_posted = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
class OTPRecord(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    otp = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.otp}"    
    
    