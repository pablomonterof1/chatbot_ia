from django.urls import path
from . import views

app_name = "bot"

urlpatterns = [
    path("", views.chat_view, name="chat"),
    path("limpiar/", views.limpiar_chat, name="limpiar_chat"),
]