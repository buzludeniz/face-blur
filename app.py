# ==================== app.py ====================
"""
Main application controller
"""

import cv2
import numpy as np
from config import Config
from modules import FaceDetector, FaceProcessor, BlurEffect, Camera


class FaceBlurApp:
    """Main application controller"""
    
    def __init__(self):
        self.camera = Camera()
        self.detector = FaceDetector()
        self.processor = FaceProcessor()
        self.blur = BlurEffect()
        
        self.frame_count = 0
        self.last_faces = []
        self.blur_types = ['box', 'gaussian', 'pixelate']
        self.current_blur_idx = 0
        
        self._print_instructions()
    
    def _print_instructions(self):
        """Print startup instructions"""
        print("Face Blur Application Started (Optimized)")
        print("Detects: Frontal + Profile faces")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'b' - Toggle blur type")
        print("  'p' - Toggle profile detection")
        print("  '+' - Increase blur")
        print("  '-' - Decrease blur")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame"""
        if self.frame_count % Config.FRAME_SKIP == 0:
            small = cv2.resize(frame, None, fx=Config.RESIZE_FOR_DETECTION, 
                             fy=Config.RESIZE_FOR_DETECTION)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            faces = self.detector.detect(gray, Config.USE_PROFILE_DETECTION)
            faces = self.processor.scale_faces(faces, Config.RESIZE_FOR_DETECTION)
            self.last_faces = self.processor.merge_overlapping(
                faces, Config.OVERLAP_THRESHOLD
            )
        
        for (x, y, w, h) in self.last_faces:
            x_p, y_p, w_p, h_p = self.processor.add_padding(
                x, y, w, h, frame.shape, Config.PADDING_RATIO
            )
            
            blurred = self.blur.apply(
                frame, x_p, y_p, w_p, h_p,
                Config.BLUR_TYPE, Config.BLUR_STRENGTH
            )
            frame[y_p:y_p+h_p, x_p:x_p+w_p] = blurred
        
        return frame
    
    def draw_info(self, frame: np.ndarray):
        """Draw info overlay"""
        info = f"{Config.BLUR_TYPE.capitalize()} ({Config.BLUR_STRENGTH}) | Faces: {len(self.last_faces)}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
        
        profile_status = "ON" if Config.USE_PROFILE_DETECTION else "OFF"
        cv2.putText(frame, f"Profiles: {profile_status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def handle_key(self, key: int) -> bool:
        """Handle keyboard input"""
        if key == ord('q'):
            return False
        elif key == ord('b'):
            self.current_blur_idx = (self.current_blur_idx + 1) % len(self.blur_types)
            Config.BLUR_TYPE = self.blur_types[self.current_blur_idx]
            print(f"Blur: {Config.BLUR_TYPE}")
        elif key == ord('p'):
            Config.USE_PROFILE_DETECTION = not Config.USE_PROFILE_DETECTION
            print(f"Profile detection {'enabled' if Config.USE_PROFILE_DETECTION else 'disabled'}")
        elif key in [ord('+'), ord('=')]:
            Config.BLUR_STRENGTH = min(199, Config.BLUR_STRENGTH + 10)
            if Config.BLUR_STRENGTH % 2 == 0:
                Config.BLUR_STRENGTH += 1
            print(f"Blur strength: {Config.BLUR_STRENGTH}")
        elif key == ord('-'):
            Config.BLUR_STRENGTH = max(11, Config.BLUR_STRENGTH - 10)
            if Config.BLUR_STRENGTH % 2 == 0:
                Config.BLUR_STRENGTH += 1
            print(f"Blur strength: {Config.BLUR_STRENGTH}")
        return True
    
    def run(self):
        """Main application loop"""
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                frame = self.process_frame(frame)
                self.draw_info(frame)
                cv2.imshow('Face Blur (Optimized)', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            print("Application closed")
