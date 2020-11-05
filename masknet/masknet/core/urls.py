from django.urls import path
from masknet.core import views
urlpatterns = [
    path('predict/', views.PredictAPI.as_view() , name="predict")
]