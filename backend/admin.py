from django.contrib import admin
from .models import UserProfile, DengueStat, HealthTip, NewsPost, MosquitoReport

# 1. User Profile & News
admin.site.register(UserProfile)
admin.site.register(NewsPost)

# 2. Dengue Stats (Detailed View)
@admin.register(DengueStat)
class DengueStatAdmin(admin.ModelAdmin):
    list_display = ('city_name', 'active_cases', 'recovered', 'deaths', 'last_updated')

# 3. ✅ Health Tips (PreventionTip ki jagah ye lagayen)
@admin.register(HealthTip)
class HealthTipAdmin(admin.ModelAdmin):
    # HealthTip model mein 'icon_class' nahi hai, 'date_posted' hai
    list_display = ('title', 'date_posted')
    
@admin.register(MosquitoReport)
class MosquitoReportAdmin(admin.ModelAdmin):
    # Yeh columns admin panel mein nazar aayenge
    list_display = ('id', 'area_name', 'status', 'created_at', 'user')
    list_filter = ('status', 'created_at')
    search_fields = ('area_name', 'description')
    readonly_fields = ('created_at',)    