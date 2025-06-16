import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import json
import time
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class AdvancedTFLiteGestureDetector:
    def __init__(self, tflite_path, labels_path, normalization_path=None, sequence_length=30, use_face=False):
        """
        Initialize the Advanced TFLite Gesture Detector with enhanced features

        Args:
            tflite_path (str): Path to the TFLite model file
            labels_path (str): Path to the JSON file containing gesture labels
            normalization_path (str): Path to normalization parameters JSON file
            sequence_length (int): Number of frames in a sequence
            use_face (bool): Whether to include face landmarks
        """
        # MediaPipe setup - matching training configuration
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # Initialize with same parameters as training
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1,
            smooth_landmarks=True
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load gesture labels
        with open(labels_path, 'r') as f:
            self.gesture_labels = json.load(f)

        # Load normalization parameters if available
        self.normalize_data = False
        self.landmarks_mean = None
        self.landmarks_std = None

        if normalization_path and os.path.exists(normalization_path):
            with open(normalization_path, 'r') as f:
                norm_data = json.load(f)
                self.landmarks_mean = np.array(norm_data['mean'])
                self.landmarks_std = np.array(norm_data['std'])
                self.normalize_data = True
                print(f"✓ Parámetros de normalización cargados desde: {normalization_path}")

        self.sequence_length = sequence_length
        self.use_face = use_face

        # Statistics tracking
        self.detection_stats = {
            'total_videos': 0,
            'successful_detections': 0,
            'failed_extractions': 0,
            'low_confidence_predictions': 0,
            'inference_times': [],
            'confidence_scores': [],
            'gesture_predictions': [],
            'video_paths': [],
            'detailed_results': []
        }

        print(f"✓ Modelo TFLite cargado: {tflite_path}")
        print(f"✓ Etiquetas cargadas: {len(self.gesture_labels)} gestos")
        print(f"✓ Forma de entrada: {self.input_details[0]['shape']}")
        print(f"✓ Forma de salida: {self.output_details[0]['shape']}")

    def extract_landmarks(self, frame):
        """Extract landmarks matching the training configuration exactly"""
        if frame is None:
            return None

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process with holistic first
        holistic_results = self.holistic.process(image)

        # Process with hands detector for better accuracy
        hands_results = self.hands.process(image)

        # If no pose landmarks detected, return None
        if not holistic_results.pose_landmarks:
            return None

        landmarks = []

        # 1. Extract KEY pose landmarks (same as training)
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

        # 2. Extract detailed hand landmarks with same logic as training
        left_hand_landmarks = None
        right_hand_landmarks = None

        # Use dedicated hands detector first (more accurate)
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                if hand_idx < len(hands_results.multi_handedness):
                    handedness = hands_results.multi_handedness[hand_idx].classification[0].label
                    if handedness == "Left":
                        left_hand_landmarks = hand_landmarks
                    elif handedness == "Right":
                        right_hand_landmarks = hand_landmarks

        # Fallback to holistic if hands not found
        if left_hand_landmarks is None and holistic_results.left_hand_landmarks:
            left_hand_landmarks = holistic_results.left_hand_landmarks

        if right_hand_landmarks is None and holistic_results.right_hand_landmarks:
            right_hand_landmarks = holistic_results.right_hand_landmarks

        # Process left hand
        if left_hand_landmarks:
            for landmark in left_hand_landmarks.landmark:
                landmarks.extend([landmark.x * 1.5, landmark.y * 1.5, landmark.z * 1.5])

                # Add derived features: distance from wrist
                wrist = left_hand_landmarks.landmark[0]
                dist_from_wrist = np.sqrt((landmark.x - wrist.x)**2 +
                                         (landmark.y - wrist.y)**2 +
                                         (landmark.z - wrist.z)**2)
                landmarks.append(dist_from_wrist)
        else:
            landmarks.extend([0.0] * 21 * 4)  # 21 points x 4 values

        # Process right hand
        if right_hand_landmarks:
            for landmark in right_hand_landmarks.landmark:
                landmarks.extend([landmark.x * 1.5, landmark.y * 1.5, landmark.z * 1.5])

                wrist = right_hand_landmarks.landmark[0]
                dist_from_wrist = np.sqrt((landmark.x - wrist.x)**2 +
                                         (landmark.y - wrist.y)**2 +
                                         (landmark.z - wrist.z)**2)
                landmarks.append(dist_from_wrist)
        else:
            landmarks.extend([0.0] * 21 * 4)

        # 3. Extract face landmarks if requested
        if self.use_face and holistic_results.face_landmarks:
            key_face_indices = [
                0, 61, 291, 78, 308, 33, 263, 133, 362, 152, 381, 10, 152, 234, 454
            ]
            for idx in key_face_indices:
                if idx < len(holistic_results.face_landmarks.landmark):
                    lm = holistic_results.face_landmarks.landmark[idx]
                    landmarks.extend([lm.x, lm.y, lm.z])

        # 4. Add relationship features between hands and pose
        if holistic_results.pose_landmarks and (left_hand_landmarks or right_hand_landmarks):
            nose = holistic_results.pose_landmarks.landmark[0]

            # Distance between hands
            if left_hand_landmarks and right_hand_landmarks:
                left_wrist = left_hand_landmarks.landmark[0]
                right_wrist = right_hand_landmarks.landmark[0]
                hand_distance = np.sqrt((left_wrist.x - right_wrist.x)**2 +
                                        (left_wrist.y - right_wrist.y)**2 +
                                        (left_wrist.z - right_wrist.z)**2)
                landmarks.append(hand_distance)
            else:
                landmarks.append(0.0)

            # Distance from each hand to nose
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
            landmarks.extend([0.0, 0.0, 0.0])

        return np.array(landmarks)

    def extract_landmark_sequence(self, video_path):
        """Extract a sequence of landmarks from a full video with improved interpolation"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Get video properties for better analysis
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.extract_landmarks(frame)
            if landmarks is not None:
                frames.append(landmarks)
            frame_count += 1

        cap.release()

        # Enhanced sequence processing with interpolation
        if len(frames) == 0:
            return None, {'original_frames': total_frames, 'extracted_frames': 0, 'fps': fps}

        # Interpolation technique matching training
        if len(frames) < self.sequence_length:
            existing_frames = len(frames)
            frames_array = np.array(frames)

            new_frames = []
            for i in range(self.sequence_length):
                frac_pos = i * (existing_frames - 1) / (self.sequence_length - 1) if self.sequence_length > 1 else 0

                idx_before = int(frac_pos)
                idx_after = min(idx_before + 1, existing_frames - 1)

                weight_after = frac_pos - idx_before
                weight_before = 1 - weight_after

                if idx_before == idx_after:
                    interpolated_frame = frames_array[idx_before]
                else:
                    interpolated_frame = frames_array[idx_before] * weight_before + frames_array[idx_after] * weight_after

                new_frames.append(interpolated_frame)

            frames = new_frames
        elif len(frames) > self.sequence_length:
            indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
            frames = [frames[i] for i in indices]

        sequence_info = {
            'original_frames': total_frames,
            'extracted_frames': len(frames),
            'fps': fps,
            'processing_method': 'interpolation' if len(frames) < self.sequence_length else 'sampling'
        }

        return np.array(frames), sequence_info

    def detect_gesture(self, video_path, confidence_threshold=0.5):
        """Detect gesture in a video with detailed analysis"""
        start_time = time.time()

        # Extract landmark sequence
        sequence, sequence_info = self.extract_landmark_sequence(video_path)

        if sequence is None:
            return {
                'gesture': 'no_gesture',
                'confidence': 0.0,
                'success': False,
                'error': 'Failed to extract landmarks',
                'sequence_info': sequence_info,
                'inference_time': time.time() - start_time
            }

        # Normalize sequence if required
        if self.normalize_data and self.landmarks_mean is not None and self.landmarks_std is not None:
            sequence = (sequence - self.landmarks_mean) / self.landmarks_std

        # Prepare input data for TFLite model
        input_data = np.array([sequence], dtype=np.float32)

        # Run inference
        inference_start = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = time.time() - inference_start

        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Get prediction details
        predictions = output_data[0]
        gesture_idx = np.argmax(predictions)
        confidence = float(predictions[gesture_idx])

        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = [
            {
                'gesture': self.gesture_labels[idx],
                'confidence': float(predictions[idx])
            }
            for idx in top_indices
        ]

        total_time = time.time() - start_time

        return {
            'gesture': self.gesture_labels[gesture_idx],
            'confidence': confidence,
            'success': confidence >= confidence_threshold,
            'top_predictions': top_predictions,
            'all_predictions': {self.gesture_labels[i]: float(predictions[i]) for i in range(len(predictions))},
            'sequence_info': sequence_info,
            'inference_time': inference_time,
            'total_time': total_time,
            'sequence_shape': sequence.shape
        }

    def test_videos_in_directory(self, test_dir, confidence_threshold=0.6, expected_gesture=None):
        """Test all videos in a directory with comprehensive analysis"""
        if not os.path.exists(test_dir):
            print(f"❌ Error: El directorio {test_dir} no existe")
            return []

        # Get all video files
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        video_files = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith(video_extensions)
        ]

        if not video_files:
            print(f"❌ No se encontraron videos en {test_dir}")
            return []

        print(f"\n🎯 INICIANDO PRUEBAS EN: {test_dir}")
        print(f"📁 Videos encontrados: {len(video_files)}")
        print(f"🎯 Umbral de confianza: {confidence_threshold}")
        if expected_gesture:
            print(f"🎭 Gesto esperado: {expected_gesture}")
        print("=" * 80)

        results = []
        gesture_counts = Counter()
        confidence_by_gesture = defaultdict(list)

        for i, video_path in enumerate(sorted(video_files)):
            video_name = os.path.basename(video_path)
            print(f"\n[{i+1}/{len(video_files)}] Procesando: {video_name}")

            result = self.detect_gesture(video_path, confidence_threshold)
            result['video_path'] = video_path
            result['video_name'] = video_name

            # Update statistics
            # These stats are accumulated across all calls to test_videos_in_directory
            # If you want stats per directory, you should reset or store them differently.
            self.detection_stats['total_videos'] += 1
            self.detection_stats['inference_times'].append(result['inference_time'])
            self.detection_stats['confidence_scores'].append(result['confidence'])
            self.detection_stats['gesture_predictions'].append(result['gesture'])
            self.detection_stats['video_paths'].append(video_path)
            self.detection_stats['detailed_results'].append(result)

            if result['success']:
                self.detection_stats['successful_detections'] += 1
                gesture_counts[result['gesture']] += 1
                confidence_by_gesture[result['gesture']].append(result['confidence'])

                print(f"  ✅ Gesto detectado: {result['gesture']}")
                print(f"  📊 Confianza: {result['confidence']:.3f}")
                print(f"  ⏱️  Tiempo total: {result['total_time']:.3f}s")
                print(f"  🧠 Tiempo inferencia: {result['inference_time']:.4f}s")

                # Show top 3 predictions
                print("  🥇 Top 3 predicciones:")
                for j, pred in enumerate(result['top_predictions']):
                    print(f"    {j+1}. {pred['gesture']}: {pred['confidence']:.3f}")

                # Check if matches expected gesture
                if expected_gesture:
                    if result['gesture'].lower() == expected_gesture.lower():
                        print(f"  ✅ Coincide con el gesto esperado")
                    else:
                        print(f"  ❌ NO coincide con el gesto esperado ({expected_gesture})")

            else:
                if result['gesture'] == 'no_gesture':
                    self.detection_stats['failed_extractions'] += 1
                    print(f"  ❌ No se pudieron extraer landmarks")
                else:
                    self.detection_stats['low_confidence_predictions'] += 1
                    print(f"  ⚠️  Confianza baja: {result['gesture']} ({result['confidence']:.3f})")

            # Show sequence information
            seq_info = result['sequence_info']
            print(f"  📹 Frames originales: {seq_info['original_frames']}")
            print(f"  🎬 Frames extraídos: {seq_info['extracted_frames']}")
            print(f"  📺 FPS: {seq_info['fps']:.1f}")

            results.append(result)

        # Generate comprehensive report for this directory
        # This call might be redundant if test_multiple_directories handles overall reporting
        # For individual directory reports, keep it; otherwise, remove.
        # self._generate_test_report(test_dir, results, gesture_counts, confidence_by_gesture, confidence_threshold, expected_gesture)

        return results

    def _generate_test_report(self, test_dir, results, gesture_counts, confidence_by_gesture, confidence_threshold, expected_gesture):
        """Generate a comprehensive test report"""
        print("\n" + "=" * 80)
        print("📊 REPORTE COMPLETO DE PRUEBAS")
        print("=" * 80)

        total_videos = len(results)
        successful = sum(1 for r in results if r['success'])
        failed_extraction = sum(1 for r in results if r['gesture'] == 'no_gesture')
        low_confidence = sum(1 for r in results if not r['success'] and r['gesture'] != 'no_gesture')

        # Overall statistics
        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"  • Total de videos: {total_videos}")
        print(f"  • Detecciones exitosas: {successful} ({successful/total_videos*100:.1f}%)")
        print(f"  • Fallos de extracción: {failed_extraction} ({failed_extraction/total_videos*100:.1f}%)")
        print(f"  • Confianza baja: {low_confidence} ({low_confidence/total_videos*100:.1f}%)")

        # Performance metrics
        inference_times_for_this_run = [r['inference_time'] for r in results]
        total_times_for_this_run = [r['total_time'] for r in results]

        if inference_times_for_this_run:
            avg_inference = np.mean(inference_times_for_this_run)
            avg_total = np.mean(total_times_for_this_run)
            print(f"\n⏱️  RENDIMIENTO:")
            print(f"  • Tiempo promedio de inferencia: {avg_inference:.4f}s")
            print(f"  • Tiempo promedio total: {avg_total:.3f}s")
            if avg_inference > 0:
                print(f"  • Inferencias por segundo: {1/avg_inference:.1f}")
            else:
                print(f"  • Inferencias por segundo: N/A (tiempo de inferencia cero)")


        # Gesture distribution
        if gesture_counts:
            print(f"\n🎭 DISTRIBUCIÓN DE GESTOS DETECTADOS:")
            for gesture, count in gesture_counts.most_common():
                avg_conf = np.mean(confidence_by_gesture[gesture])
                print(f"  • {gesture}: {count} videos (confianza promedio: {avg_conf:.3f})")

        # Confidence analysis
        confidences = [r['confidence'] for r in results if r['confidence'] > 0]
        if confidences:
            print(f"\n📊 ANÁLISIS DE CONFIANZA:")
            print(f"  • Confianza promedio: {np.mean(confidences):.3f}")
            print(f"  • Confianza mínima: {np.min(confidences):.3f}")
            print(f"  • Confianza máxima: {np.max(confidences):.3f}")
            print(f"  • Desviación estándar: {np.std(confidences):.3f}")

        # Accuracy if expected gesture is provided
        if expected_gesture:
            correct_predictions = sum(1 for r in results if r['success'] and r['gesture'].lower() == expected_gesture.lower())
            accuracy = correct_predictions / total_videos * 100 if total_videos > 0 else 0
            print(f"\n🎯 PRECISIÓN PARA GESTO '{expected_gesture}':")
            print(f"  • Predicciones correctas: {correct_predictions}/{total_videos}")
            print(f"  • Precisión: {accuracy:.1f}%")

        # Failed cases analysis
        failed_cases = [r for r in results if not r['success']]
        if failed_cases:
            print(f"\n❌ ANÁLISIS DE FALLOS:")
            print(f"  • Videos con fallos: {len(failed_cases)}")

            # Group by failure type
            extraction_failures = [r for r in failed_cases if r['gesture'] == 'no_gesture']
            confidence_failures = [r for r in failed_cases if r['gesture'] != 'no_gesture']

            if extraction_failures:
                print(f"  • Fallos de extracción de landmarks: {len(extraction_failures)}")
                for r in extraction_failures[:3]:  # Show first 3
                    print(f"    - {r['video_name']}: {r['error']}")

            if confidence_failures:
                print(f"  • Confianza baja (< {confidence_threshold}): {len(confidence_failures)}")
                for r in confidence_failures[:3]:  # Show first 3
                    print(f"    - {r['video_name']}: {r['gesture']} ({r['confidence']:.3f})")

        # Save detailed report to file
        self._save_detailed_report(test_dir, results)

    def _save_detailed_report(self, test_dir, results):
        """Save detailed report to CSV and JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"test_reports_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)

        # Save to CSV
        csv_data = []
        for r in results:
            csv_data.append({
                'video_name': r['video_name'],
                'gesture': r['gesture'],
                'confidence': r['confidence'],
                'success': r['success'],
                'inference_time': r['inference_time'],
                'total_time': r['total_time'],
                'original_frames': r['sequence_info']['original_frames'],
                'extracted_frames': r['sequence_info']['extracted_frames'],
                'fps': r['sequence_info']['fps']
            })

        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(report_dir, f"test_results_{os.path.basename(test_dir)}.csv")
        df.to_csv(csv_path, index=False)

        # Save detailed JSON
        json_path = os.path.join(report_dir, f"detailed_results_{os.path.basename(test_dir)}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 REPORTES GUARDADOS:")
        print(f"  • CSV: {csv_path}")
        print(f"  • JSON: {json_path}")

    def test_multiple_directories(self, base_dir, confidence_threshold=0.6):
        """Test multiple directories, each representing a different gesture"""
        if not os.path.exists(base_dir):
            print(f"❌ Error: El directorio base {base_dir} no existe")
            return

        gesture_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        if not gesture_dirs:
            print(f"❌ No se encontraron subdirectorios en {base_dir}")
            return

        print(f"\n🚀 PRUEBAS MASIVAS EN: {base_dir}")
        print(f"📂 Directorios encontrados: {len(gesture_dirs)}")
        print("=" * 100)

        all_results = {}
        overall_stats = {
            'total_videos': 0,
            'total_correct': 0,
            'total_successful': 0,
            'total_failed_extractions': 0,
            'total_low_confidence': 0,
            'all_inference_times': [],
            'all_total_times': []
        }

        # This will store aggregated gesture counts for confusion matrix
        aggregated_predictions = defaultdict(Counter)


        for gesture_dir in sorted(gesture_dirs):
            full_path = os.path.join(base_dir, gesture_dir)
            print(f"\n🎭 Probando gesto: {gesture_dir}")

            # Reset internal stats for each directory's detailed report, but accumulate for overall
            # (The existing self.detection_stats accumulates globally, which is useful for overall averages)
            current_directory_results = self.test_videos_in_directory(full_path, confidence_threshold, gesture_dir)
            all_results[gesture_dir] = current_directory_results

            if current_directory_results:
                correct = sum(1 for r in current_directory_results if r['success'] and r['gesture'].lower() == gesture_dir.lower())
                successful = sum(1 for r in current_directory_results if r['success'])
                failed_extraction_dir = sum(1 for r in current_directory_results if r['gesture'] == 'no_gesture')
                low_confidence_dir = sum(1 for r in current_directory_results if not r['success'] and r['gesture'] != 'no_gesture')


                overall_stats['total_videos'] += len(current_directory_results)
                overall_stats['total_correct'] += correct
                overall_stats['total_successful'] += successful
                overall_stats['total_failed_extractions'] += failed_extraction_dir
                overall_stats['total_low_confidence'] += low_confidence_dir
                overall_stats['all_inference_times'].extend([r['inference_time'] for r in current_directory_results])
                overall_stats['all_total_times'].extend([r['total_time'] for r in current_directory_results])


                print(f"  📊 Resumen para {gesture_dir}:")
                print(f"    • Videos: {len(current_directory_results)}")
                print(f"    • Correctos: {correct} ({correct/len(current_directory_results)*100:.1f}%)")
                print(f"    • Exitosos (confianza alta): {successful} ({successful/len(current_directory_results)*100:.1f}%)")
                print(f"    • Fallos de extracción: {failed_extraction_dir}")
                print(f"    • Confianza baja: {low_confidence_dir}")

                # Populate aggregated_predictions for confusion matrix
                for res in current_directory_results:
                    if res['success']:
                        aggregated_predictions[gesture_dir][res['gesture']] += 1
                    else:
                        aggregated_predictions[gesture_dir]['No Detection/Low Confidence'] += 1 # Or a more specific category


        # Overall summary
        print("\n" + "=" * 100)
        print("🏆 RESUMEN GENERAL DE TODAS LAS PRUEBAS")
        print("=" * 100)

        if overall_stats['total_videos'] > 0:
            overall_accuracy = overall_stats['total_correct'] / overall_stats['total_videos'] * 100
            overall_success_rate = overall_stats['total_successful'] / overall_stats['total_videos'] * 100

            print(f"📊 Total de videos procesados: {overall_stats['total_videos']}")
            print(f"🎯 Predicciones correctas (que coinciden con el directorio): {overall_stats['total_correct']} ({overall_accuracy:.1f}%)")
            print(f"✅ Detecciones exitosas (con confianza >= {confidence_threshold}): {overall_stats['total_successful']} ({overall_success_rate:.1f}%)")
            print(f"❌ Fallos de extracción de landmarks: {overall_stats['total_failed_extractions']}")
            print(f"⚠️  Predicciones con confianza baja: {overall_stats['total_low_confidence']}")

            # Overall performance metrics
            if overall_stats['all_inference_times']:
                overall_avg_inference = np.mean(overall_stats['all_inference_times'])
                overall_avg_total_time = np.mean(overall_stats['all_total_times'])
                print(f"\n⏱️  RENDIMIENTO PROMEDIO GENERAL:")
                print(f"  • Tiempo promedio de inferencia: {overall_avg_inference:.4f}s")
                print(f"  • Tiempo promedio total por video: {overall_avg_total_time:.3f}s")
                if overall_avg_inference > 0:
                    print(f"  • Inferencias por segundo: {1/overall_avg_inference:.1f}")
                else:
                    print(f"  • Inferencias por segundo: N/A (tiempo de inferencia cero)")


            # Confusion matrix-like analysis
            print(f"\n🔍 MATRIZ DE CONFUSIÓN (Esperado vs. Detectado):")
            # Collect all unique detected gestures for header
            all_detected_labels = set()
            for expected_gesture, detected_counts in aggregated_predictions.items():
                all_detected_labels.update(detected_counts.keys())
            all_detected_labels = sorted(list(all_detected_labels))

            # Header
            header = f"{'Esperado':<20}" + "".join(f"{label:<20}" for label in all_detected_labels) + f"{'Total':<10}"
            print(header)
            print("-" * len(header))

            # Rows
            for expected_gesture in sorted(gesture_dirs): # Iterate through expected gestures (directories)
                row_str = f"{expected_gesture:<20}"
                total_videos_in_dir = len(all_results.get(expected_gesture, [])) # Get total for current expected gesture
                
                for detected_label in all_detected_labels:
                    count = aggregated_predictions[expected_gesture].get(detected_label, 0)
                    row_str += f"{count:<20}"
                row_str += f"{total_videos_in_dir:<10}" # Add total for the row
                print(row_str)

        return all_results

def main():
    """Main function to test TFLite models with comprehensive analysis"""
    # Configuration
    output_dir = 'res_optimized'  # Adjust to your output directory
    tflite_path = os.path.join(output_dir, 'gesture_classification_model.tflite')
    labels_path = os.path.join(output_dir, 'gesture_labels.json')
    normalization_path = os.path.join(output_dir, 'normalization_params.json')

    # Verify if files exist
    if not os.path.exists(tflite_path):
        print(f"❌ Error: No se encontró el modelo TFLite en {tflite_path}")
        return

    if not os.path.exists(labels_path):
        print(f"❌ Error: No se encontraron las etiquetas en {labels_path}")
        return

    # Initialize detector
    detector = AdvancedTFLiteGestureDetector(
        tflite_path=tflite_path,
        labels_path=labels_path,
        normalization_path=normalization_path,
        sequence_length=30,
        use_face=False  # Change to True if you used face landmarks in training
    )

    # Example Usage:
    # 1. Test videos in a single directory, expecting a specific gesture
    # single_test_dir = 'data/test_videos/wave' # Example path
    # print(f"\n--- Ejecutando prueba para un solo directorio: {single_test_dir} ---")
    # detector.test_videos_in_directory(single_test_dir, confidence_threshold=0.7, expected_gesture='wave')

    # 2. Test videos across multiple directories, where each directory name is the expected gesture
    base_test_data_dir = 'prueba_final/' # Root directory containing gesture subfolders (e.g., data/test_videos/wave, data/test_videos/clapping)
    print(f"\n--- Ejecutando pruebas masivas en múltiples directorios: {base_test_data_dir} ---")
    detector.test_multiple_directories(base_test_data_dir, confidence_threshold=0.6)

if __name__ == "__main__":
    main()