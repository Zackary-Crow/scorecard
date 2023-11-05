from django.urls import path
from dev import views
from django.conf.urls.static import static
from django.conf import settings
from backend import consumers


urlpatterns = [
    path("",views.HomeView.as_view()),
    path("endpoint",views.api, name = "api")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

websocket_urlpatterns = [
    path("ws/camera",consumers.CameraConsumer.as_asgi())
]