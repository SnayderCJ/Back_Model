import os
import cv2
import numpy as np
import pandas as pd
from django.conf import settings
from tensorflow.python.keras.models import load_model
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from .models import Sample, Training
from .utils import (
    create_folder,
    read_frames_from_directory,
    normalize_frames,
    save_normalized_frames,
    get_keypoints,
    clear_directory
)
from mediapipe.python.solutions.holistic import Holistic
from .model import get_model

# Vista para renderizar el template de captura de video
class VideoCaptureView(APIView):
    def get(self, request):
        return render(request, 'video_capture.html')

# Vista para predecir la acción basada en los keypoints generados
class PredictActionView(APIView):
    def post(self, request, sample_id):
        try:
            # Cargar la muestra correspondiente
            sample = Sample.objects.get(id=sample_id)

            # Cargar el archivo de keypoints
            if not sample.keypoints_file or not os.path.exists(sample.keypoints_file):
                return Response({'error': 'Keypoints no encontrados.'}, status=status.HTTP_404_NOT_FOUND)

            # Cargar los keypoints desde el archivo .h5
            keypoints_sequence = pd.read_hdf(sample.keypoints_file, key='keypoints').to_numpy()
            keypoints_sequence = keypoints_sequence.reshape(1, settings.MODEL_FRAMES, -1)  # Ajustar la forma para el modelo

            # Cargar el modelo entrenado desde el archivo .keras
            model_file_path = os.path.join(settings.MODEL_FOLDER_PATH, 'actions_15.keras')  # Asegúrate de tener esta ruta
            if not os.path.exists(model_file_path):
                return Response({'error': 'Modelo no encontrado.'}, status=status.HTTP_404_NOT_FOUND)

            try:
                model = load_model(model_file_path)
            except Exception as e:
                return Response({'error': f'Error al cargar el modelo: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Realizar la predicción con el modelo
            prediction = model.predict(keypoints_sequence)
            predicted_class = np.argmax(prediction)

            # Retornar la clase predicha
            return Response({'predicted_class': int(predicted_class)}, status=status.HTTP_200_OK)

        except Sample.DoesNotExist:
            return Response({'error': 'Sample no encontrado.'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Vista para capturar muestras y guardarlas en FRAME_ACTIONS_PATH
class CaptureSamplesView(APIView):
    def post(self, request):
        label = request.data.get('label')
        video_file = request.FILES.get('video')
        
        if not label or not video_file:
            return Response({'error': 'El label y el archivo de video son requeridos.'}, status=status.HTTP_400_BAD_REQUEST)

        # Ruta donde se guardarán los frames
        frames_dir = os.path.join(settings.FRAME_ACTIONS_PATH, label)
        create_folder(frames_dir)

        # Capturar y guardar los frames del video en frames_dir
        video_capture = cv2.VideoCapture(video_file.temporary_file_path())
        frame_count = 0
        success, frame = video_capture.read()

        while success:
            frame_path = os.path.join(frames_dir, f'frame_{frame_count:04}.jpg')
            cv2.imwrite(frame_path, frame)
            success, frame = video_capture.read()
            frame_count += 1
        
        video_capture.release()

        # Crear la instancia de Sample
        sample = Sample.objects.create(frames_directory=frames_dir, label=label)
        return Response({'message': 'Muestras capturadas exitosamente.', 'sample_id': sample.id}, status=status.HTTP_201_CREATED)

# Vista para normalizar muestras
class NormalizeSamplesView(APIView):
    def post(self, request, sample_id):
        try:
            sample = Sample.objects.get(id=sample_id)
            frames = read_frames_from_directory(sample.frames_directory)
            normalized_frames = normalize_frames(frames, target_frame_count=settings.MODEL_FRAMES)

            # Ruta para los frames normalizados
            normalized_dir = f'{sample.frames_directory}_normalized'
            create_folder(normalized_dir)
            save_normalized_frames(normalized_dir, normalized_frames)

            sample.normalized_frames_directory = normalized_dir
            sample.save()

            return Response({'message': 'Muestras normalizadas exitosamente.'}, status=status.HTTP_200_OK)
        except Sample.DoesNotExist:
            return Response({'error': 'Sample no encontrado.'}, status=status.HTTP_404_NOT_FOUND)

# Vista para generar keypoints
class CreateKeypointsView(APIView):
    def post(self, request, sample_id):
        try:
            sample = Sample.objects.get(id=sample_id)
            holistic = Holistic(static_image_mode=True)
            keypoints_sequence = get_keypoints(holistic, sample.normalized_frames_directory)

            # Ruta para guardar los keypoints
            keypoints_file = os.path.join(settings.KEYPOINTS_PATH, f'{sample.label}_keypoints.h5')
            create_folder(settings.KEYPOINTS_PATH)  # Crear la carpeta si no existe
            pd.DataFrame(keypoints_sequence).to_hdf(keypoints_file, key='keypoints', mode='w')

            sample.keypoints_file = keypoints_file
            sample.save()

            return Response({'message': 'Keypoints generados exitosamente.'}, status=status.HTTP_200_OK)
        except Sample.DoesNotExist:
            return Response({'error': 'Sample no encontrado.'}, status=status.HTTP_404_NOT_FOUND)

# Vista para entrenar el modelo
class TrainModelView(APIView):
    def post(self, request):
        # Obtener parámetros de entrenamiento
        max_length_frames = settings.MODEL_FRAMES
        output_length = request.data.get('output_length', 10)
        
        # Cargar datos de entrenamiento
        # Aquí cargarías los datos a partir de los archivos .h5 generados anteriormente

        # Definir y entrenar el modelo
        model = get_model(max_length_frames, output_length)
        # Ajuste del modelo con los datos de entrenamiento
        # model.fit(...)

        # Guardar el modelo entrenado en la ruta especificada en settings
        model_file_path = settings.MODEL_PATH
        create_folder(settings.MODEL_FOLDER_PATH)  # Crear la carpeta del modelo si no existe
        model.save(model_file_path)

        training = Training.objects.create(model_file=model_file_path)
        return Response({'message': 'Modelo entrenado exitosamente.', 'training_id': training.id}, status=status.HTTP_201_CREATED)
