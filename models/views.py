import os
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from django.conf import settings
from keras import models
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
    
class PrediccionCaptureView(APIView):
    def get(self, request):
        return render(request, 'prediccion.html')

# Vista para predecir la acción basada en los keypoints generados
class PredictActionView(APIView):
    def post(self, request):
        try:
            # Recibir landmarks desde el frontend
            landmarks = request.data.get('landmarks')

            if not landmarks:
                return Response({'error': 'Landmarks no encontrados.'}, status=status.HTTP_400_BAD_REQUEST)

            # Procesar los landmarks
            keypoints_sequence = np.array([[point['x'], point['y'], point['z']] for point in landmarks])

            # Ajustar el tamaño de keypoints_sequence
            if keypoints_sequence.shape[0] < settings.MODEL_FRAMES:
                keypoints_sequence = np.pad(
                    keypoints_sequence,
                    ((0, settings.MODEL_FRAMES - keypoints_sequence.shape[0]), (0, 0)),
                    'constant'
                )
            elif keypoints_sequence.shape[0] > settings.MODEL_FRAMES:
                keypoints_sequence = keypoints_sequence[:settings.MODEL_FRAMES]

            # Cambiar la forma de los keypoints para que coincida con lo que espera el modelo
            keypoints_sequence = keypoints_sequence.reshape(1, settings.MODEL_FRAMES, settings.LENGTH_KEYPOINTS)

            # Cargar el modelo
            model_file_path = os.path.join(settings.MODEL_FOLDER_PATH, f'actions_{settings.MODEL_FRAMES}.h5')
            if not os.path.exists(model_file_path):
                return Response({'error': 'Modelo no encontrado.'}, status=status.HTTP_404_NOT_FOUND)

            model = models.load_model(model_file_path, compile=False)

            # Realizar la predicción
            prediction = model.predict(keypoints_sequence)
            predicted_class_index = np.argmax(prediction)

            # Obtener el nombre de la clase predicha desde la base de datos
            predicted_sample = Sample.objects.filter(id=predicted_class_index).first()
            if not predicted_sample:
                return Response({'error': 'Clase predicha no encontrada en la base de datos.'}, status=status.HTTP_404_NOT_FOUND)

            predicted_class_name = predicted_sample.label
            return Response({'predicted_word': predicted_class_name}, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response({'error': f'Error de valor: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            # Log del error para depuración adicional
            import traceback
            print("Error inesperado:", traceback.format_exc())
            return Response({'error': f'Error inesperado: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



    # Vista para capturar muestras y guardarlas en FRAME_ACTIONS_PATH
class CaptureSamplesView(APIView):
    def post(self, request):
        label = request.data.get('label')
        video_file = request.FILES.get('video')
        
        
        if not label or not video_file:
            return Response({'error': 'El label y el archivo de video son requeridos.'}, status=status.HTTP_400_BAD_REQUEST)

        # Ruta donde se guardarán los frames
        frames_dir = os.path.join(settings.FRAME_ACTIONS_PATH, label)
        os.makedirs(frames_dir, exist_ok=True)

        # Guardar el video temporalmente en el sistema de archivos
        temp_video_path = os.path.join(frames_dir, 'temp_video.webm')
        with open(temp_video_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # Cargar el video utilizando OpenCV
        video_capture = cv2.VideoCapture(temp_video_path)

        frame_count = 0
        success, frame = video_capture.read()

        while success:
            frame_path = os.path.join(frames_dir, f'frame_{frame_count:04}.jpg')
            cv2.imwrite(frame_path, frame)
            success, frame = video_capture.read()
            frame_count += 1
        
        video_capture.release()

        # Eliminar el archivo temporal después de extraer los frames
        os.remove(temp_video_path)

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
            normalized_dir = f'{sample.frames_directory}'
            clear_directory(normalized_dir)
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
            keypoints_file = os.path.join(settings.KEYPOINTS_PATH, f'{sample.label}.h5')
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
        
        # Entrenar el modelo con los datos cargados
        # model.fit(...)

        # Crear la carpeta del modelo si no existe
        create_folder(settings.MODEL_FOLDER_PATH)
        
        # Definir la ruta para guardar el modelo en formato HDF5
        model_file_path = os.path.join(settings.MODEL_FOLDER_PATH, 'actions_15.h5')  # Cambia el nombre si es necesario
        
        # Guardar el modelo entrenado en formato HDF5
        model.save(model_file_path, save_format='h5')  # Especificar el formato
        
        # Crear una entrada en la base de datos para el entrenamiento
        training = Training.objects.create(model_file=model_file_path)
        
        return Response({'message': 'Modelo entrenado exitosamente.', 'training_id': training.id}, status=status.HTTP_201_CREATED)

