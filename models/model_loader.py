# model_loader.py
from tensorflow.python.keras.models import load_model
from django.conf import settings
import os

# Ruta del modelo
MODEL_PATH = settings.MODEL_PATH

# Cargar el modelo desde el archivo .keras
model  = load_model(MODEL_PATH)
