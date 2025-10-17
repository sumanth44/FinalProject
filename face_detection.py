import cv2
import numpy as np
import face_recognition
import base64
from PIL import Image
import io
import json
from typing import List, Dict, Optional, Tuple

class FaceDetectionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_face_phones = []
        
    def load_known_faces(self, users_data: List[Dict]):
        """Load known faces from database"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_face_phones = []
        
        for user in users_data:
            if user.get('embedding'):
                self.known_face_encodings.append(np.array(user['embedding']))
                self.known_face_names.append(user['name'])
                self.known_face_ids.append(user['user_id'])
                self.known_face_phones.append(user['phone'])
        
        print(f"Loaded {len(self.known_face_encodings)} known faces for recognition")
    
    def add_new_face(self, user_data: Dict):
        """Add a new face to the known faces list"""
        if user_data.get('embedding'):
            self.known_face_encodings.append(np.array(user_data['embedding']))
            self.known_face_names.append(user_data['name'])
            self.known_face_ids.append(user_data['user_id'])
            self.known_face_phones.append(user_data['phone'])
            print(f"Added new face: {user_data['name']} (ID: {user_data['user_id']})")
    
    def retrain_system(self, users_data: List[Dict]):
        """Retrain the entire face recognition system with updated data"""
        print("Retraining face recognition system...")
        self.load_known_faces(users_data)
        print("Face recognition system retrained successfully")
    
    def detect_faces_in_image(self, image_data: str) -> List[Dict]:
        """Detect and recognize faces in base64 encoded image"""
        try:
            print("Starting face detection...")
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            
            print(f"Image size: {image_rgb.shape}")
            
            # Try multiple face detection methods
            face_locations = []
            
            # Method 1: HOG model (faster)
            try:
                face_locations = face_recognition.face_locations(image_rgb, model="hog")
                print(f"HOG model found {len(face_locations)} faces")
            except Exception as e:
                print(f"HOG model failed: {e}")
            
            # Method 2: CNN model (more accurate) if HOG fails
            if len(face_locations) == 0:
                try:
                    face_locations = face_recognition.face_locations(image_rgb, model="cnn")
                    print(f"CNN model found {len(face_locations)} faces")
                except Exception as e:
                    print(f"CNN model failed: {e}")
            
            # Method 3: OpenCV Haar cascades as fallback
            if len(face_locations) == 0:
                try:
                    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    print(f"OpenCV found {len(faces)} faces")
                    
                    # Convert OpenCV format to face_recognition format
                    for (x, y, w, h) in faces:
                        face_locations.append((y, x + w, y + h, x))
                except Exception as e:
                    print(f"OpenCV detection failed: {e}")
            
            if len(face_locations) == 0:
                print("No faces detected with any method")
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
            print(f"Generated {len(face_encodings)} face encodings")
            
            results = []
            
            if len(self.known_face_encodings) == 0:
                print("No known faces loaded for comparison")
                return []
            
            print(f"Comparing with {len(self.known_face_encodings)} known faces")
            
            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                print(f"Processing face {i+1}...")
                
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=0.6  # Slightly higher tolerance for better detection
                )
                
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                
                print(f"Best match distance: {best_distance:.3f}, threshold: 0.6")
                
                # Use a more lenient threshold for recognition
                if matches[best_match_index] and best_distance < 0.6:
                    # Face recognized
                    confidence = max(0, 1 - best_distance)
                    results.append({
                        'user_id': self.known_face_ids[best_match_index],
                        'name': self.known_face_names[best_match_index],
                        'phone': self.known_face_phones[best_match_index],
                        'confidence': round(confidence, 3),
                        'face_location': face_location
                    })
                    print(f"Face recognized: {self.known_face_names[best_match_index]} (confidence: {confidence:.3f})")
                else:
                    # Unknown face
                    results.append({
                        'user_id': None,
                        'name': 'Unknown',
                        'phone': None,
                        'confidence': 0,
                        'face_location': face_location
                    })
                    print(f"Unknown face detected (best distance: {best_distance:.3f})")
            
            print(f"Face detection completed. Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detect and recognize faces in video frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Try multiple face detection methods
            face_locations = []
            
            # Method 1: HOG model (faster for video)
            try:
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            except Exception as e:
                print(f"HOG model failed in video: {e}")
            
            # Method 2: OpenCV Haar cascades as fallback
            if len(face_locations) == 0:
                try:
                    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    # Convert OpenCV format to face_recognition format
                    for (x, y, w, h) in faces:
                        face_locations.append((y, x + w, y + h, x))
                except Exception as e:
                    print(f"OpenCV detection failed in video: {e}")
            
            if len(face_locations) == 0:
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            results = []
            
            if len(self.known_face_encodings) == 0:
                return []
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=0.6  # Slightly higher tolerance for video
                )
                
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                
                # Use a more lenient threshold for recognition
                if matches[best_match_index] and best_distance < 0.6:
                    # Face recognized
                    confidence = max(0, 1 - best_distance)
                    results.append({
                        'user_id': self.known_face_ids[best_match_index],
                        'name': self.known_face_names[best_match_index],
                        'phone': self.known_face_phones[best_match_index],
                        'confidence': round(confidence, 3),
                        'face_location': face_location
                    })
                else:
                    # Unknown face
                    results.append({
                        'user_id': None,
                        'name': 'Unknown',
                        'phone': None,
                        'confidence': 0,
                        'face_location': face_location
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def draw_face_boxes(self, frame: np.ndarray, face_results: List[Dict]) -> np.ndarray:
        """Draw bounding boxes around detected faces"""
        for result in face_results:
            face_location = result['face_location']
            top, right, bottom, left = face_location
            
            # Choose color based on recognition status
            if result['user_id']:
                color = (0, 255, 0)  # Green for recognized
                label = f"{result['name']} ({result['confidence']:.2f})"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = "Unknown"
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw label text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
        
        return frame

# Global face detection system instance
face_detector = FaceDetectionSystem()
