from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from django.conf import settings

def get_model(max_length_frames, output_length: int):
    # Usar LENGTH_KEYPOINTS del archivo de configuración de settings
    length_keypoints = settings.LENGTH_KEYPOINTS
    
    model = Sequential()
    
    model.add(LSTM(64, return_sequences=True, input_shape=(length_keypoints,max_length_frames), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(output_length, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
