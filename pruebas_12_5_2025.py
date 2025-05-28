import os
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from collections import deque
import time
import random
from sklearn.model_selection import train_test_split

# Importación para metadata_writer
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer
from mediapipe.tasks.metadata.metadata_schema_py_generated import TensorMetadataT, ContentT, ImagePropertiesT, FeaturePropertiesT, ProcessUnitT, AssociatedFileT, ModelMetadataT
from mediapipe.tasks.python.metadata import metadata

class GestureDetector:
    def __init__(self, sequence_length=30, use_face=False):
        # Configuración de MediaPipe y parámetros
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
        self.gesture_model = None
        self.gesture_labels = []
        self.sequence_length = sequence_length  # Número de frames a capturar por secuencia
        self.use_face = use_face  # Opción para incluir landmarks faciales
        
        # Configuración holística con mayor confianza para mejor precisión
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1,  # Usar complejidad media (0-2)
            smooth_landmarks=True  # Suavizado de landmarks para reducir ruido
        )
        
        # Detector de manos independiente para mayor precisión
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # Directorios para guardar modelos y datos
        self.output_dir = 'res_optimized'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Rutas de archivos
        self.model_path = os.path.join(self.output_dir, 'gesture_classification_model.h5')
        self.tflite_path = os.path.join(self.output_dir, 'gesture_classification_model.tflite')
        self.quant_tflite_path = os.path.join(self.output_dir, 'gesture_classification_model_quant.tflite')
        self.labels_path = os.path.join(self.output_dir, 'gesture_labels.json')
        self.task_path = os.path.join(self.output_dir, 'gesture_classification_model.task')
        
        # Variables para normalización
        self.normalize_data = True
        self.landmarks_mean = None
        self.landmarks_std = None

    def extract_landmarks(self, frame):
        """Extrae landmarks de pose, manos usando MediaPipe con mayor enfoque en manos"""
        if frame is None:
            return None
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Procesar con holistic primero
        holistic_results = self.holistic.process(image)
        
        # Procesar con detector de manos independiente para mayor precisión
        hands_results = self.hands.process(image)
        
        # Si no se detectan landmarks fundamentales, retornar None
        if not holistic_results.pose_landmarks:
            return None
            
        landmarks = []
        
        # 1. Extraer landmarks CLAVE de pose (33 puntos x 4 valores: x, y, z, visibility)
        # Tomamos solo 25 landmarks clave de pose en lugar de todos
        key_pose_indices = [
            0,   # nariz
            11, 12, 13, 14, 15, 16,  # hombros, codos, muñecas
            23, 24, 25, 26, 27, 28,  # caderas, rodillas, tobillos
            5, 6,  # ojos
            9, 10   # orejas
        ]
        
        if holistic_results.pose_landmarks:
            for idx in key_pose_indices:
                lm = holistic_results.pose_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # 2. Extraer landmarks detallados de manos
        # Usar el detector de manos dedicado para mayor precisión si está disponible
        left_hand_landmarks = None
        right_hand_landmarks = None
        
        # Primero verificar las manos del detector independiente (más preciso)
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                if hand_idx < len(hands_results.multi_handedness):
                    handedness = hands_results.multi_handedness[hand_idx].classification[0].label
                    if handedness == "Left":
                        left_hand_landmarks = hand_landmarks
                    elif handedness == "Right":
                        right_hand_landmarks = hand_landmarks
        
        # Si no se encontraron manos con el detector independiente, usar holistic como respaldo
        if left_hand_landmarks is None and holistic_results.left_hand_landmarks:
            left_hand_landmarks = holistic_results.left_hand_landmarks
            
        if right_hand_landmarks is None and holistic_results.right_hand_landmarks:
            right_hand_landmarks = holistic_results.right_hand_landmarks
        
        # Procesar mano izquierda si está disponible
        if left_hand_landmarks:
            # Dar más peso a los landmarks de manos (multiplicando por 1.5)
            for landmark in left_hand_landmarks.landmark:
                # Añadir características adicionales para manos: velocidad de movimiento o curvatura
                landmarks.extend([landmark.x * 1.5, landmark.y * 1.5, landmark.z * 1.5])
                
                # Añadir características derivadas: distancia desde la muñeca al landmark
                wrist = left_hand_landmarks.landmark[0]  # landmark 0 es la muñeca
                dist_from_wrist = np.sqrt((landmark.x - wrist.x)**2 + 
                                         (landmark.y - wrist.y)**2 + 
                                         (landmark.z - wrist.z)**2)
                landmarks.append(dist_from_wrist)
        else:
            # Si no hay mano izquierda, añadir ceros
            landmarks.extend([0.0] * 21 * 4)  # 21 puntos x 4 valores (x,y,z,dist)
            
        # Procesar mano derecha si está disponible
        if right_hand_landmarks:
            for landmark in right_hand_landmarks.landmark:
                landmarks.extend([landmark.x * 1.5, landmark.y * 1.5, landmark.z * 1.5])
                
                # Añadir características derivadas: distancia desde la muñeca
                wrist = right_hand_landmarks.landmark[0]
                dist_from_wrist = np.sqrt((landmark.x - wrist.x)**2 + 
                                         (landmark.y - wrist.y)**2 + 
                                         (landmark.z - wrist.z)**2)
                landmarks.append(dist_from_wrist)
        else:
            # Si no hay mano derecha, añadir ceros
            landmarks.extend([0.0] * 21 * 4)  # 21 puntos x 4 valores
        
        # 3. Extraer landmarks faciales SOLO si se solicita (opcional)
        if self.use_face and holistic_results.face_landmarks:
            # Seleccionar solo landmarks faciales clave (15 puntos) para no sobrecargar
            key_face_indices = [
                0,   # nariz
                61, 291,  # labios
                78, 308,  # cejas
                33, 263,  # mejillas
                133, 362,  # ojos
                152, 381,  # párpados
                10, 152,   # frente
                234, 454   # mandíbula
            ]
            for idx in key_face_indices:
                if idx < len(holistic_results.face_landmarks.landmark):
                    lm = holistic_results.face_landmarks.landmark[idx]
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        # 4. Añadir características de relación entre manos y pose
        if holistic_results.pose_landmarks and (left_hand_landmarks or right_hand_landmarks):
            nose = holistic_results.pose_landmarks.landmark[0]  # Nariz
            
            # Distancia entre manos
            if left_hand_landmarks and right_hand_landmarks:
                left_wrist = left_hand_landmarks.landmark[0]
                right_wrist = right_hand_landmarks.landmark[0]
                hand_distance = np.sqrt((left_wrist.x - right_wrist.x)**2 + 
                                      (left_wrist.y - right_wrist.y)**2 + 
                                      (left_wrist.z - right_wrist.z)**2)
                landmarks.append(hand_distance)
            else:
                landmarks.append(0.0)
                
            # Distancia de cada mano a la nariz
            if left_hand_landmarks:
                left_wrist = left_hand_landmarks.landmark[0]
                left_to_nose = np.sqrt((left_wrist.x - nose.x)**2 + 
                                      (left_wrist.y - nose.y)**2 + 
                                      (left_wrist.z - nose.z)**2)
                landmarks.append(left_to_nose)
            else:
                landmarks.append(0.0)
                
            if right_hand_landmarks:
                right_wrist = right_hand_landmarks.landmark[0]
                right_to_nose = np.sqrt((right_wrist.x - nose.x)**2 + 
                                       (right_wrist.y - nose.y)**2 + 
                                       (right_wrist.z - nose.z)**2)
                landmarks.append(right_to_nose)
            else:
                landmarks.append(0.0)
        else:
            landmarks.extend([0.0, 0.0, 0.0])  # Relaciones entre manos/pose
            
        return np.array(landmarks)
        
    def extract_landmark_sequence(self, video_path):
        """Extrae una secuencia de landmarks de un video completo con mejor manejo de secuencias"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            landmarks = self.extract_landmarks(frame)
            if landmarks is not None:
                frames.append(landmarks)
                
        cap.release()
        
        # Procesar la secuencia para que tenga longitud fija
        if len(frames) == 0:
            return None
            
        # Técnica de interpolación mejorada para secuencias
        if len(frames) < self.sequence_length:
            # Usar interpolación para generar nuevos frames
            existing_frames = len(frames)
            frames_array = np.array(frames)
            
            new_frames = []
            for i in range(self.sequence_length):
                # Calcular posición fraccional en la secuencia original
                frac_pos = i * (existing_frames - 1) / (self.sequence_length - 1) if self.sequence_length > 1 else 0
                
                # Obtener índices para interpolación
                idx_before = int(frac_pos)
                idx_after = min(idx_before + 1, existing_frames - 1)
                
                # Calcular peso de interpolación
                weight_after = frac_pos - idx_before
                weight_before = 1 - weight_after
                
                # Interpolar linealmente
                if idx_before == idx_after:
                    interpolated_frame = frames_array[idx_before]
                else:
                    interpolated_frame = frames_array[idx_before] * weight_before + frames_array[idx_after] * weight_after
                
                new_frames.append(interpolated_frame)
            
            frames = new_frames
        # Si hay demasiados frames, hacer un muestreo uniforme
        elif len(frames) > self.sequence_length:
            indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
            frames = [frames[i] for i in indices]
            
        return np.array(frames)

    def list_videos_in_directories(self, base_dir):
        """Lista todos los videos en directorios, agrupados por gesto"""
        gesture_videos = {}
        for gesture_dir in os.listdir(base_dir):
            gesture_path = os.path.join(base_dir, gesture_dir)
            if os.path.isdir(gesture_path):
                video_files = [
                    os.path.join(gesture_path, f)
                    for f in os.listdir(gesture_path)
                    if f.lower().endswith(('.mp4', '.avi', '.mov'))
                ]
                if video_files:
                    gesture_videos[gesture_dir] = video_files
        return gesture_videos

    def prepare_training_data(self, gesture_videos):
        """Prepara los datos de entrenamiento con normalización y aumento de datos"""
        X_sequences = []
        y_train = []
        gesture_labels = list(gesture_videos.keys())
        
        print("Preparando datos de entrenamiento...")
        for idx, (gesture, videos) in enumerate(gesture_videos.items()):
            print(f"Procesando gesto: {gesture} ({len(videos)} videos)")
            for video_path in videos:
                sequence = self.extract_landmark_sequence(video_path)
                if sequence is not None:
                    X_sequences.append(sequence)
                    y_train.append(idx)
                    
                    # Técnicas de aumento de datos para mejorar robustez
                    # 1. Añadir secuencia con pequeñas variaciones aleatorias
                    noise_factor = 0.005  # Factor de ruido pequeño
                    noisy_sequence = sequence + np.random.normal(0, noise_factor, sequence.shape)
                    X_sequences.append(noisy_sequence)
                    y_train.append(idx)
                    
                    # 2. Añadir secuencia con velocidad ligeramente diferente (más lenta o más rápida)
                    if len(sequence) >= 5:  # Solo si hay suficientes frames
                        speed_factor = random.uniform(0.9, 1.1)  # 10% más lento o más rápido
                        new_length = int(len(sequence) * speed_factor)
                        new_length = max(5, min(new_length, len(sequence) * 2))  # Limitar el cambio
                        
                        indices = np.linspace(0, len(sequence) - 1, new_length)
                        resampled_sequence = np.array([sequence[int(i)] if i < len(sequence) else sequence[-1] for i in indices])
                        
                        # Asegurar que tenga la longitud correcta
                        if len(resampled_sequence) > self.sequence_length:
                            step = len(resampled_sequence) / self.sequence_length
                            indices = [int(i * step) for i in range(self.sequence_length)]
                            resampled_sequence = resampled_sequence[indices]
                        elif len(resampled_sequence) < self.sequence_length:
                            # Rellenar con repetición del último frame
                            padding = self.sequence_length - len(resampled_sequence)
                            resampled_sequence = np.vstack([resampled_sequence, np.tile(resampled_sequence[-1], (padding, 1))])
                            
                        X_sequences.append(resampled_sequence)
                        y_train.append(idx)
        
        if not X_sequences:
            return [], [], []
                
        # Convertir a array numpy
        X_data = np.array(X_sequences)
        y_data = np.array(y_train)
        
        # Normalizar los datos para mejor generalización
        if self.normalize_data:
            # Calcular media y desviación estándar
            self.landmarks_mean = np.mean(X_data, axis=(0, 1))
            self.landmarks_std = np.std(X_data, axis=(0, 1))
            self.landmarks_std[self.landmarks_std == 0] = 1  # Evitar división por cero
            
            # Normalizar datos
            X_data = (X_data - self.landmarks_mean) / self.landmarks_std
            
            # Guardar normalización para uso posterior
            normalization_data = {
                'mean': self.landmarks_mean.tolist(),
                'std': self.landmarks_std.tolist()
            }
            norm_path = os.path.join(self.output_dir, 'normalization_params.json')
            with open(norm_path, 'w') as f:
                json.dump(normalization_data, f)
            print(f"Parámetros de normalización guardados en: {norm_path}")
                
        return X_data, y_data, gesture_labels

    def build_advanced_model(self, input_shape, num_classes):
        """Construye un modelo CNN+LSTM mejorado para reconocimiento de gestos"""
        model = Sequential()
        
        # Capa convolucional temporal para extraer patrones locales
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        
        # Segunda capa convolucional
        model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        
        # Capa LSTM bidireccional para capturar dependencias temporales
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.3))
        
        # Segunda capa LSTM para secuencias más complejas
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        
        # Capas densas para clasificación
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_gesture_model(self, base_dir):
        """Entrena un modelo para reconocer secuencias de gestos con optimizaciones."""
        print(f"Iniciando entrenamiento con videos de: {base_dir}")
        gesture_videos = self.list_videos_in_directories(base_dir)

        if not gesture_videos:
            print(f"No se encontraron videos en {base_dir}")
            return []

        X_data, y_data, gesture_labels = self.prepare_training_data(gesture_videos)
        self.gesture_labels = gesture_labels

        if len(X_data) == 0:
            print("No se pudieron extraer secuencias de los videos")
            return []

        print(f"Forma de los datos de entrenamiento: {X_data.shape}")
        print(f"Número de clases: {len(gesture_labels)}")

        # Dividir datos en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )

        # Convertir etiquetas a formato categorical
        y_train_categorical = to_categorical(y_train, num_classes=len(gesture_labels))
        y_val_categorical = to_categorical(y_val, num_classes=len(gesture_labels))

        # Obtener la forma de entrada para el modelo
        input_shape = (X_train.shape[1], X_train.shape[2])

        # Construir y compilar el modelo mejorado
        self.gesture_model = self.build_advanced_model(input_shape, len(gesture_labels))
        print(self.gesture_model.summary())

        # Configurar callbacks para entrenamiento
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Reducción de learning rate cuando el entrenamiento se estanca
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        # Entrenar el modelo
        print("Entrenando modelo...")
        history = self.gesture_model.fit(
            X_train, y_train_categorical,
            epochs=100,
            batch_size=16,
            validation_data=(X_val, y_val_categorical),
            callbacks=[early_stopping, reduce_lr]
        )

        # Evaluar el modelo
        val_loss, val_acc = self.gesture_model.evaluate(X_val, y_val_categorical)
        print(f"Precisión en validación: {val_acc:.4f}")

        # Guardar el modelo entrenado
        self.gesture_model.save(self.model_path)
        print(f"Modelo guardado en: {self.model_path}")

        # Convertir a TFLite (versión normal)
        self._convert_to_tflite()
        
        # Convertir a TFLite con cuantización para reducir tamaño
        self._convert_to_tflite_quantized()

        # Guardar etiquetas
        with open(self.labels_path, 'w') as f:
            json.dump(gesture_labels, f)
        print(f"Etiquetas guardadas en: {self.labels_path}")

        # Crear metadatos y archivo task
        self.create_task_file()

        return gesture_labels
    
    def _convert_to_tflite(self):
        """Convierte el modelo a TFLite con optimizaciones para móviles"""
        print("Convirtiendo modelo a TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.gesture_model)
        
        # Configurar optimizaciones
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Usar punto flotante en lugar de cuantización para máxima precisión
        converter.target_spec.supported_types = [tf.float32]
        
        # Configuraciones para mejorar rendimiento en Android
        converter._experimental_lower_tensor_list_ops = False
        
        # Convertir modelo
        tflite_model = converter.convert()
        
        # Guardar modelo TFLite
        with open(self.tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Modelo TFLite guardado en: {self.tflite_path}")
        
        # Comprobar tamaño del modelo
        model_size_mb = os.path.getsize(self.tflite_path) / (1024 * 1024)
        print(f"Tamaño del modelo TFLite: {model_size_mb:.2f} MB")
    
    def _convert_to_tflite_quantized(self):
        """Convierte el modelo a TFLite con cuantización para reducir tamaño"""
        print("Convirtiendo modelo a TFLite con cuantización...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.gesture_model)
        
        # Optimizaciones
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Definir una función para generar un conjunto de datos representativo
        # Esto es necesario para la cuantización completa de enteros
        def representative_dataset_gen():
            # Generar algunos ejemplos artificiales basados en la forma de entrada del modelo
            # No necesitamos datos reales, solo ejemplos con la forma correcta para calibración
            input_shape = self.gesture_model.input_shape
            # Generar 100 muestras aleatorias para calibración
            for _ in range(100):
                # Si tenemos datos disponibles de entrenamiento, usarlos
                if hasattr(self, 'X_train_sample') and self.X_train_sample is not None:
                    idx = np.random.randint(0, len(self.X_train_sample))
                    sample = self.X_train_sample[idx:idx+1]
                else:
                    # Si no hay datos disponibles, generar datos aleatorios con la forma correcta
                    # Usamos una distribución normal para simular datos normalizados
                    sample = np.random.normal(0, 1, size=(1, input_shape[1], input_shape[2]))
                
                # Asegurar que es float32
                yield [sample.astype(np.float32)]
        
        # Configurar dataset representativo
        converter.representative_dataset = representative_dataset_gen
        
        # Para dispositivos compatibles con operaciones INT8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        # Opcional: forzar cuantización completa (entrada y salida también)
        # Comentamos estas líneas para mantener entrada/salida como float32
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
        
        # Convertir modelo
        try:
            tflite_quant_model = converter.convert()
            
            # Guardar modelo TFLite cuantizado
            with open(self.quant_tflite_path, 'wb') as f:
                f.write(tflite_quant_model)
            print(f"Modelo TFLite cuantizado guardado en: {self.quant_tflite_path}")
            
            # Comprobar tamaño del modelo
            model_size_mb = os.path.getsize(self.quant_tflite_path) / (1024 * 1024)
            print(f"Tamaño del modelo TFLite cuantizado: {model_size_mb:.2f} MB")
        except Exception as e:
            print(f"Error en la cuantización: {e}")
            print("Continuando sin modelo cuantizado...")

    def detect_gesture(self, video_path):
        """Detecta el gesto en un video usando el modelo entrenado"""
        if self.gesture_model is None:
            try:
                from tensorflow.keras.models import load_model
                self.gesture_model = load_model(self.model_path)
                with open(self.labels_path, 'r') as f:
                    self.gesture_labels = json.load(f)
                    
                # Cargar parámetros de normalización si existen
                norm_path = os.path.join(self.output_dir, 'normalization_params.json')
                if os.path.exists(norm_path):
                    with open(norm_path, 'r') as f:
                        norm_data = json.load(f)
                        self.landmarks_mean = np.array(norm_data['mean'])
                        self.landmarks_std = np.array(norm_data['std'])
                        self.normalize_data = True
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
                return "no_gesture", 0.0
                
        # Extraer secuencia de landmarks del video
        sequence = self.extract_landmark_sequence(video_path)
        
        if sequence is None:
            return "no_gesture", 0.0
            
        # Normalizar secuencia si es necesario
        if self.normalize_data and self.landmarks_mean is not None and self.landmarks_std is not None:
            sequence = (sequence - self.landmarks_mean) / self.landmarks_std
            
        # Realizar predicción
        pred = self.gesture_model.predict(np.array([sequence]), verbose=0)
        gesture_idx = np.argmax(pred[0])
        confidence = pred[0][gesture_idx]
        
        return self.gesture_labels[gesture_idx], float(confidence)
    
    def create_task_file(self):
        """Crea un archivo .task de MediaPipe a partir del modelo TFLite con metadatos."""
        try:
            # 1. Leer el modelo TFLite (usamos versión normalizada por defecto)
            with open(self.tflite_path, 'rb') as model_file:
                model_buffer = model_file.read()

            # 2. Crear un escritor de metadatos para el modelo
            writer = metadata_writer.MetadataWriter.create(model_buffer)

            # Agregar información general del modelo
            writer.add_general_info(
                model_name="GestureClassifierAdvanced",
                model_description="Clasificador de gestos optimizado para uso móvil"
            )

            # Agregar metadatos de entrada para landmarks
            writer.add_feature_input(
                name="landmark_input",
                description="Secuencia de landmarks corporales para reconocimiento de gestos"
            )

            # Preparar las etiquetas para la salida de clasificación
            labels_txt_path = os.path.join(self.output_dir, "labels.txt")
            with open(labels_txt_path, 'w') as f:
                for label in self.gesture_labels:
                    f.write(f"{label}\n")

            # Crear objeto Labels y añadir desde el archivo
            label_object = metadata_writer.Labels().add_from_file(
                label_filepath=labels_txt_path
            )

            # Agregar metadatos de salida para clasificación
            writer.add_classification_output(
                labels=label_object,
                score_calibration=None,
                name="gesture_prediction",
                description="Probabilidades para cada gesto reconocido"
            )

            # 3. Poblar el modelo con los metadatos
            tflite_with_metadata, _ = writer.populate()

            # 4. Empaquetar el modelo TFLite con metadatos en un archivo .task
            populator = metadata.MetadataPopulator.with_model_buffer(tflite_with_metadata)

            # Verificar si los archivos ya están empaquetados
            packed_files = populator.get_packed_associated_file_list() or []
            if 'labels.txt' not in packed_files:
                populator.load_associated_files([labels_txt_path])

            # Asegúrate de que el modelo TFLite esté incluido y registrado
            populator.populate()
            task_model_buffer = populator.get_model_buffer()

            # 5. Guardar el modelo final como archivo .task
            with open(self.task_path, 'wb') as task_file:
                task_file.write(task_model_buffer)

            print(f"Archivo .task creado exitosamente en: {self.task_path}")
            print(f"Archivo labels.txt creado en: {labels_txt_path}")

            # También guardar versión cuantizada como .task si existe
            if os.path.exists(self.quant_tflite_path):
                quant_task_path = os.path.join(self.output_dir, 'gesture_classification_model_quant.task')
                
                with open(self.quant_tflite_path, 'rb') as model_file:
                    quant_model_buffer = model_file.read()
                
                quant_writer = metadata_writer.MetadataWriter.create(quant_model_buffer)
                quant_writer.add_general_info(
                    model_name="GestureClassifierAdvancedQuantized",
                    model_description="Clasificador de gestos optimizado y cuantizado para uso móvil"
                )
                quant_writer.add_feature_input(
                    name="landmark_input",
                    description="Secuencia de landmarks corporales para reconocimiento de gestos"
                )
                quant_writer.add_classification_output(
                    labels=label_object,
                    name="gesture_prediction",
                    description="Probabilidades para cada gesto reconocido"
                )
                
                tflite_quant_with_metadata, _ = quant_writer.populate()
                quant_populator = metadata.MetadataPopulator.with_model_buffer(tflite_quant_with_metadata)
                
                if 'labels.txt' not in (quant_populator.get_packed_associated_file_list() or []):
                    quant_populator.load_associated_files([labels_txt_path])
                
                quant_populator.populate()
                quant_task_model_buffer = quant_populator.get_model_buffer()
                
                with open(quant_task_path, 'wb') as task_file:
                    task_file.write(quant_task_model_buffer)
                    
                print(f"Archivo .task cuantizado creado exitosamente en: {quant_task_path}")

        except Exception as e:
            print(f"Error al crear archivo task: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    # Inicializar detector de gestos
    detector = GestureDetector(sequence_length=30)

    # Entrenar modelo con los videos
    gesture_labels = detector.train_gesture_model('nuevos_videos')
    print("Gestos entrenados:", gesture_labels)

    # Probar con videos de prueba
    test_videos = ["prueba_felipe/20250428_014224.mp4",
"prueba_felipe/20250514104912.mp4",
"prueba_felipe/2.mp4", 
"prueba_felipe/1.mp4", 
"prueba_felipe/2.mp4", 
"prueba_felipe/3.mp4", 
"prueba_felipe/4.mp4", 
"prueba_felipe/5.mp4", 
"prueba_felipe/6.mp4",
"prueba_felipe/20250428_014218.mp4",
"prueba_felipe/20250428_014237.mp4",
"prueba_felipe/20250428_014241.mp4",
"prueba_felipe/20250428_014934.mp4",
"prueba_felipe/20250504_221438.mp4",
"prueba_felipe/202505141049.mp4",
"prueba_felipe/2025051410491.mp4",
"prueba_felipe/2025051410492.mp4",
"prueba_felipe/2025051410493.mp4",
"prueba_felipe/20250514104911.mp4",
"prueba_felipe/20250514104913.mp4",
"prueba_felipe/20250514104914.mp4",
"prueba_felipe/20250514104921.mp4",
"prueba_felipe/20250514104922.mp4",
"prueba_felipe/20250514104923.mp4",
"prueba_felipe/20250514104931.mp4",
"prueba_felipe/20250514104932.mp4",
"prueba_felipe/21mayo.mp4",
"prueba_felipe/22mayo.mp4",
"prueba_felipe/23mayo.mp4",
"prueba_felipe/24mayo.mp4",
"prueba_felipe/25mayo.mp4",
"prueba_felipe/26mayo.mp4",
"prueba_felipe/27mayo.mp4",
"prueba_felipe/28mayo.mp4",
"prueba_felipe/29mayo.mp4",
"prueba_felipe/30mayo.mp4",
"prueba_felipe/31mayo.mp4",
"prueba_felipe/32mayo.mp4",
"prueba_felipe/33mayo.mp4",
"prueba_felipe/34mayo.mp4",
"prueba_felipe/35mayo.mp4",
"prueba_felipe/36mayo.mp4",
"prueba_felipe/37mayo.mp4",
"prueba_felipe/Febrero 1.mp4",
"prueba_felipe/Febrero 2.mp4",
"prueba_felipe/Febrero 3.mp4",
"prueba_felipe/Febrero 4.mp4",
"prueba_felipe/Febrero 5.mp4",
"prueba_felipe/Febrero 6.mp4",
"prueba_felipe/Febrero 7.mp4",
"prueba_felipe/Febrero 8.mp4",
"prueba_felipe/Febrero 9.mp4",
"prueba_felipe/Febrero 10.mp4",
"prueba_felipe/Febrero 11.mp4",
"prueba_felipe/Febrero 12.mp4",
"prueba_felipe/Febrero 13.mp4",
"prueba_felipe/Febrero 14.mp4",
"prueba_felipe/Febrero 15.mp4",
"prueba_felipe/Febrero 16.mp4",
"prueba_felipe/Febrero 17.mp4",
"prueba_felipe/Febrero 18.mp4",
"prueba_felipe/Febrero 19.mp4",
"prueba_felipe/Febrero 20.mp4",
"prueba_felipe/Febrero 21.mp4",
"prueba_felipe/Febrero 21.mp4",
"prueba_felipe/prima1.mp4",
"prueba_felipe/prima2.mp4",
"prueba_felipe/prima3.mp4",
"prueba_felipe/prima4.mp4",
"prueba_felipe/prima5.mp4",
"prueba_felipe/prima6.mp4",
"prueba_felipe/prima7.mp4",
"prueba_felipe/prima8.mp4",
"prueba_felipe/prima9.mp4",
"prueba_felipe/prima10.mp4",
"prueba_felipe/prima11.mp4",
"prueba_felipe/prima12.mp4",
"prueba_felipe/prima13.mp4",
"prueba_felipe/prima14.mp4",
"prueba_felipe/prima15.mp4",
"prueba_felipe/prima16.mp4",
"prueba_felipe/Prima_2.mp4",
"prueba_felipe/Prima_3.mp4",
"prueba_felipe/Prima_4.mp4",
"prueba_felipe/Prima_5.mp4",
"prueba_felipe/primo1.mp4",
"prueba_felipe/primo2.mp4",
"prueba_felipe/primo3.mp4",
"prueba_felipe/primo4.mp4",
"prueba_felipe/primo5.mp4",
"prueba_felipe/primo6.mp4",
"prueba_felipe/primo7.mp4",
"prueba_felipe/primo8.mp4",
"prueba_felipe/primo9.mp4",
"prueba_felipe/primo10.mp4",
"prueba_felipe/primo11.mp4",
"prueba_felipe/primo12.mp4",
"prueba_felipe/primo13.mp4",
"prueba_felipe/primo14.mp4",
"prueba_felipe/primo15.mp4",
"prueba_felipe/primo16.mp4",
"prueba_felipe/hija 1.mp4",
"prueba_felipe/hija 2.mp4",
"prueba_felipe/hija 3.mp4",
"prueba_felipe/hija 4.mp4",
"prueba_felipe/hija 5.mp4",
"prueba_felipe/hija 6.mp4",
"prueba_felipe/hija 7.mp4",
"prueba_felipe/hija 8.mp4",
"prueba_felipe/hija 9.mp4",
"prueba_felipe/hija 10.mp4",
"prueba_felipe/hija 11.mp4",
"prueba_felipe/hija 12.mp4",
"prueba_felipe/hija 13.mp4",
"prueba_felipe/hija 14.mp4",
"prueba_felipe/hija 15.mp4",
"prueba_felipe/hija 16.mp4",
"prueba_felipe/hija 17.mp4",
"prueba_felipe/hija 18.mp4",
"prueba_felipe/hija 19.mp4",
"prueba_felipe/hija 20.mp4",
"prueba_felipe/hija 21.mp4",
"prueba_felipe/hija 22.mp4",
"prueba_felipe/hija 23.mp4",
"prueba_felipe/hija 24.mp4",
"prueba_felipe/hija 25.mp4",
"prueba_felipe/hija 26.mp4",
"prueba_felipe/hija 27.mp4",
"prueba_felipe/hija 28.mp4",
"prueba_felipe/hija 29.mp4",
"prueba_felipe/hija 30.mp4",
"prueba_felipe/hija 31.mp4",
"prueba_felipe/hija 32.mp4",
"prueba_felipe/hija 33.mp4",
"prueba_felipe/hija 34.mp4",
"prueba_felipe/hija 35.mp4",
"prueba_felipe/hija 36.mp4",
"prueba_felipe/hija 37.mp4",
"prueba_felipe/hijo 1.mp4",
"prueba_felipe/hijo 2.mp4",
"prueba_felipe/hijo 3.mp4",
"prueba_felipe/hijo 4.mp4",
"prueba_felipe/hijo 5.mp4",
"prueba_felipe/hijo 6.mp4",
"prueba_felipe/hijo 7.mp4",
"prueba_felipe/hijo 8.mp4",
"prueba_felipe/hijo 9.mp4",
"prueba_felipe/hijo 10.mp4",
"prueba_felipe/hijo 11.mp4",
"prueba_felipe/hijo 12.mp4",
"prueba_felipe/hijo 13.mp4",
"prueba_felipe/hijo 14.mp4",
"prueba_felipe/hijo 15.mp4",
"prueba_felipe/hijo 16.mp4",
"prueba_felipe/hijo 17.mp4",
"prueba_felipe/hijo 18.mp4",
"prueba_felipe/hijo 19.mp4",
"prueba_felipe/hijo 20.mp4",
"prueba_felipe/hijo 21.mp4",
"prueba_felipe/hijo 22.mp4",
"prueba_felipe/hijo 23.mp4",
"prueba_felipe/hijo 24.mp4",
"prueba_felipe/hijo 25.mp4",
"prueba_felipe/hijo 26.mp4",
"prueba_felipe/hijo 27.mp4",
"prueba_felipe/hijo 28.mp4",
"prueba_felipe/hijo 29.mp4",
"prueba_felipe/hijo 30.mp4",
"prueba_felipe/hijo 31.mp4",
"prueba_felipe/hijo 32.mp4",
"prueba_felipe/hijo 33.mp4",
"prueba_felipe/hijo 34.mp4",
"prueba_felipe/hijo 35.mp4",
"prueba_felipe/hijo 36.mp4",
"prueba_felipe/hijo 37.mp4",
"prueba_felipe/1Nuera.mp4",
"prueba_felipe/2Nuera.mp4",
"prueba_felipe/3Nuera.mp4",
"prueba_felipe/4Nuera.mp4",
"prueba_felipe/5Nuera.mp4",
"prueba_felipe/6Nuera.mp4",
"prueba_felipe/7Nuera.mp4",
"prueba_felipe/8Nuera.mp4",
"prueba_felipe/9Nuera.mp4",
"prueba_felipe/10Nuera.mp4",
"prueba_felipe/11Nuera.mp4",
"prueba_felipe/12Nuera.mp4",
"prueba_felipe/13Nuera.mp4",
"prueba_felipe/14Nuera.mp4",
"prueba_felipe/15Nuera.mp4",
"prueba_felipe/16Nuera.mp4",
"prueba_felipe/17Nuera.mp4",
"prueba_felipe/18Nuera.mp4",
"prueba_felipe/19Nuera.mp4",
"prueba_felipe/20Nuera.mp4",
"prueba_felipe/21Nuera.mp4",
"prueba_felipe/22Nuera.mp4",
"prueba_felipe/23Nuera.mp4",
"prueba_felipe/24Nuera.mp4",
"prueba_felipe/25Nuera.mp4",
"prueba_felipe/26Nuera.mp4",
"prueba_felipe/27Nuera.mp4",
"prueba_felipe/28Nuera.mp4",
"prueba_felipe/29Nuera.mp4",
"prueba_felipe/30Nuera.mp4",
"prueba_felipe/31Nuera.mp4",
"prueba_felipe/32Nuera.mp4",
"prueba_felipe/33Nuera.mp4",
"prueba_felipe/34Nuera.mp4",
"prueba_felipe/35Nuera.mp4",
"prueba_felipe/1mes.mp4",
"prueba_felipe/2mes.mp4",
"prueba_felipe/3mes.mp4",
"prueba_felipe/4mes.mp4",
"prueba_felipe/5mes.mp4",
"prueba_felipe/6mes.mp4",
"prueba_felipe/7mes.mp4",
"prueba_felipe/8mes.mp4",
"prueba_felipe/9mes.mp4",
"prueba_felipe/10mes.mp4",
"prueba_felipe/11mes.mp4",
"prueba_felipe/12mes.mp4",
"prueba_felipe/13mes.mp4",
"prueba_felipe/14mes.mp4",
"prueba_felipe/15mes.mp4",
"prueba_felipe/16mes.mp4",
"prueba_felipe/17mes.mp4",
"prueba_felipe/18mes.mp4",
"prueba_felipe/19mes.mp4",
"prueba_felipe/20mes.mp4",
"prueba_felipe/21mes.mp4",
"prueba_felipe/22mes.mp4",
"prueba_felipe/23mes.mp4",
"prueba_felipe/24mes.mp4",
"prueba_felipe/25mes.mp4",
"prueba_felipe/26mes.mp4",
"prueba_felipe/27mes.mp4",
"prueba_felipe/28mes.mp4",
"prueba_felipe/29mes.mp4",
"prueba_felipe/30mes.mp4",
"prueba_felipe/31mes.mp4",
"prueba_felipe/32mes.mp4",
"prueba_felipe/33mes.mp4",
"prueba_felipe/34mes.mp4",
"prueba_felipe/35mes.mp4",
]

    for test_video in test_videos:
        gesture, confidence = detector.detect_gesture(test_video)
        if confidence > 0.6:
            print(f"Gesto detectado en {test_video}: {gesture} (Confianza: {confidence:.2f})")
        else:
            print(f"Gesto no reconocido en {test_video}")

if __name__ == "__main__":
    main()