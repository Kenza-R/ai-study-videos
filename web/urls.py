from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path
from .views import home, health, static_debug, upload_paper, pipeline_status, pipeline_result, register

urlpatterns = [
    path("", home, name="home"),
    path("health", health, name="health"),
    path("static-debug/", static_debug, name="static_debug"),
    path("upload/", upload_paper, name="upload_paper"),
    path("status/<str:pmid>/", pipeline_status, name="pipeline_status"),
    path("result/<str:pmid>/", pipeline_result, name="pipeline_result"),
    path("login/", LoginView.as_view(template_name="registration/login.html"), name="login"),
    path("logout/", LogoutView.as_view(next_page="/"), name="logout"),
    path("register/", register, name="register"),
]
