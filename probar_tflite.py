import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import json

class TFLiteGestureDetector:
    def __init__(self, tflite_path, labels_path, sequence_length=30):
        """
        Initialize the TFLite Gesture Detector
        
        Args:
            tflite_path (str): Path to the TFLite model file
            labels_path (str): Path to the JSON file containing gesture labels
            sequence_length (int, optional): Number of frames in a sequence. Defaults to 30.
        """
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
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
        
        self.sequence_length = sequence_length

    def extract_landmarks(self, frame):
        """Extract landmarks from a single frame using MediaPipe Holistic"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        
        # If no pose landmarks detected, return None
        if not results.pose_landmarks:
            return None
            
        landmarks = []
        
        # Extract pose landmarks (33 points x 4 values: x, y, z, visibility)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        # Extract left hand landmarks (21 points x 3 values: x, y, z)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            # If no left hand, add zeros
            landmarks.extend([0.0] * 21 * 3)
            
        # Extract right hand landmarks (21 points x 3 values: x, y, z)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            # If no right hand, add zeros
            landmarks.extend([0.0] * 21 * 3)
            
        return np.array(landmarks)

    def extract_landmark_sequence(self, video_path):
        """Extract a sequence of landmarks from a full video"""
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
        
        # Process the sequence to have a fixed length
        if len(frames) == 0:
            return None
            
        # If too few frames, duplicate some to reach sequence_length
        if len(frames) < self.sequence_length:
            ratio = self.sequence_length / len(frames)
            new_frames = []
            for i in range(self.sequence_length):
                idx = min(int(i / ratio), len(frames) - 1)
                new_frames.append(frames[idx])
            frames = new_frames
        # If too many frames, do uniform sampling
        elif len(frames) > self.sequence_length:
            indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
            frames = [frames[i] for i in indices]
            
        return np.array(frames)

    def detect_gesture(self, video_path):
        """Detect gesture in a video using the TFLite model"""
        # Extract landmark sequence
        sequence = self.extract_landmark_sequence(video_path)
        
        if sequence is None:
            return "no_gesture", 0.0
        
        # Prepare input data for TFLite model
        input_data = np.array([sequence], dtype=np.float32)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Get prediction
        gesture_idx = np.argmax(output_data[0])
        confidence = output_data[0][gesture_idx]
        
        return self.gesture_labels[gesture_idx], float(confidence)

def main():
    # Paths to model files (adjust these to your specific paths)
    output_dir = 'res_felipe2'
    tflite_path = os.path.join(output_dir, 'gesture_classification_model.tflite')
    labels_path = os.path.join(output_dir, 'gesture_labels.json')
    
    # Initialize TFLite Gesture Detector
    detector = TFLiteGestureDetector(tflite_path, labels_path)

    # Test videos directory
    test_videos_dir = 'prueba_felipe'
    
    # Get all test videos
    test_videos = [
        os.path.join(test_videos_dir, f) 
        for f in os.listdir(test_videos_dir) 
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ]

    # Test each video
    for test_video in test_videos:
        gesture, confidence = detector.detect_gesture(test_video)
        if confidence > 0.6:
            print(f"Gesture detected in {test_video}: {gesture} (Confidence: {confidence:.2f})")
        else:
            print(f"No recognized gesture in {test_video}")

if __name__ == "__main__":
    main()

