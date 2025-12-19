# ==================== modules/detector.py ====================
"""
Face detection module
"""

import cv2
import numpy as np
from config import Config


class FaceDetector:
    """Handles face detection with multiple cascades"""
    
    def __init__(self):
        self.frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.frontal_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        self.profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
    
    def detect(self, gray: np.ndarray, use_profiles: bool = True) -> np.ndarray:
        """Detect all faces including profiles"""
        faces = []
        
        # Primary frontal detection
        frontal = self.frontal.detectMultiScale(
            gray, Config.SCALE_FACTOR, Config.MIN_NEIGHBORS,
            minSize=Config.MIN_FACE_SIZE, flags=cv2.CASCADE_SCALE_IMAGE
        )
        faces.extend(frontal)
        
        if use_profiles:
            # Profile detection (right side)
            profile_r = self.profile.detectMultiScale(
                gray, Config.SCALE_FACTOR, Config.MIN_NEIGHBORS - 1,
                minSize=Config.MIN_FACE_SIZE, flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces.extend(profile_r)
            
            # Profile detection (left side - flip image)
            gray_flip = cv2.flip(gray, 1)
            profile_l = self.profile.detectMultiScale(
                gray_flip, Config.SCALE_FACTOR, Config.MIN_NEIGHBORS - 1,
                minSize=Config.MIN_FACE_SIZE, flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert flipped coordinates back
            for (x, y, w, h) in profile_l:
                faces.append([gray.shape[1] - x - w, y, w, h])
        
        return np.array(faces) if faces else np.array([])
