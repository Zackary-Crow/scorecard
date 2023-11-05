from django.urls import re_path, path, include
from . import consumers

websocket_urlpatterns= [
    path("ws/camera",consumers.CameraConsumer.as_asgi())
]