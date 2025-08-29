#!/usr/bin/env python3
"""
AVer CAM520 Pro Controller - Rebuilt using HTTP method from sniffedtest.py
"""
import sys
import cv2
import numpy as np
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox,
                             QComboBox, QCheckBox, QSpinBox, QGridLayout, QLineEdit,
                             QSplitter, QMessageBox, QFormLayout)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - face tracking disabled")

from camera_controller import HTTPCameraController
from simple_camera import get_all_cameras, VideoCapture
from virtual_webcam import VirtualWebcam


class FaceTracker:
    """Face detection using MediaPipe"""
    
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.min_confidence = 0.5  # Default 50% confidence
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=self.min_confidence)
        else:
            self.face_detection = None
    
    def set_confidence_threshold(self, confidence):
        """Update face detection confidence threshold"""
        if MEDIAPIPE_AVAILABLE and confidence != self.min_confidence:
            self.min_confidence = confidence
            # Recreate the face detection with new confidence
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=self.min_confidence)
            print(f"FaceTracker: Confidence threshold updated to {confidence:.2f}")
    
    def detect_faces(self, image):
        """Detect faces and return normalized coordinates - with timeout protection"""
        if not self.face_detection:
            return []
        
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Face detection timeout")
            
            # Set timeout for face detection (Windows compatible version)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image for faster processing if it's large
            h, w = rgb_image.shape[:2]
            if w > 640:
                scale = 640 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                rgb_image = cv2.resize(rgb_image, (new_w, new_h))
            
            results = self.face_detection.process(rgb_image)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    faces.append({
                        'x': bbox.xmin,
                        'y': bbox.ymin,
                        'width': bbox.width,
                        'height': bbox.height,
                        'confidence': detection.score[0],
                        'center_x': bbox.xmin + bbox.width / 2,
                        'center_y': bbox.ymin + bbox.height / 2
                    })
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []


class TrackingController:
    """Face tracking with HTTP PTZ control - fluid movement using direct positioning"""
    
    def __init__(self, camera_controller):
        self.camera_controller = camera_controller
        self.deadzone = 0.08  # Reasonable deadzone for definitive movement
        self.tightness = 0.3
        self.last_movement_time = 0
        self.tracking_frequency = 0.1   # 100ms for responsive tracking
        
        # Position tracking for smooth movement
        self.current_pan = 0    # Current pan position in degrees
        self.current_tilt = 0   # Current tilt position in degrees  
        self.current_zoom = 100  # Current zoom value
        
        # Definitive tracking parameters
        self.min_movement_degrees = 1.0    # Only move if >= 1 degree correction needed
        self.last_significant_movement = 0  # Track when we last made a significant move
        
        # Last known face position tracking
        self.last_known_face = None
        self.face_lost_time = 0
        self.face_lost_timeout = 2.0  # Continue tracking for 2 seconds after face is lost
        self.last_tracking_position = None  # Store last calculated movement
    
    def set_deadzone(self, deadzone):
        # Convert from percentage and make sure it's not too large
        self.deadzone = min(0.15, deadzone)  # Cap at 15% to ensure responsiveness
        print(f"Deadzone set to: {self.deadzone:.3f} (input: {deadzone:.3f})")
    
    def set_tightness(self, tightness):
        self.tightness = tightness
        print(f"TrackingController: Tightness updated to: {tightness:.3f} (this affects auto-zoom target size)")
    
    def set_frequency(self, frequency):
        """Set tracking update frequency (in seconds)"""
        self.tracking_frequency = frequency
        print(f"TrackingController: Frequency set to {frequency:.1f}s")
    
    def _start_smooth_tracking(self):
        """Start smooth continuous tracking"""
        if self.is_smoothing:
            return
            
        self.is_smoothing = True
        self._smooth_step()
    
    def _smooth_step(self):
        """Execute one step of smooth tracking"""
        if not self.is_smoothing:
            return
        
        # Check if we're close enough to target
        if abs(self.target_deviation_x) < self.deadzone and abs(self.target_deviation_y) < self.deadzone:
            self.is_smoothing = False
            return
        
        # Calculate movement direction and intensity
        pan_dir = 0
        tilt_dir = 0
        
        if abs(self.target_deviation_x) > self.deadzone:
            pan_dir = 1 if self.target_deviation_x > 0 else -1
        
        if abs(self.target_deviation_y) > self.deadzone:
            tilt_dir = 1 if self.target_deviation_y > 0 else -1  # Revert Y direction back to normal
        
        # Use very short movements for smoothness
        if pan_dir != 0 or tilt_dir != 0:
            # Calculate duration based on deviation magnitude
            deviation_mag = max(abs(self.target_deviation_x), abs(self.target_deviation_y))
            duration = max(0.03, min(0.08, deviation_mag * 0.1))  # 30-80ms movements
            
            self.camera_controller.pan_tilt(pan_dir, tilt_dir, duration)
            print(f"Smooth step: pan={pan_dir}, tilt={tilt_dir}, duration={duration:.3f}s, targets=({self.target_deviation_x:.3f},{self.target_deviation_y:.3f})")
        
        # Schedule next step
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(60, self._smooth_step)  # 60ms between steps
    
    def update_tracking(self, faces):
        """Fluid face tracking with proper direction handling"""
        import time
        
        print(f"\n=== TRACKING UPDATE CALLED ===")
        print(f"Faces detected: {len(faces) if faces else 0}")
        
        current_time = time.time()
        time_since_last = current_time - self.last_movement_time
        print(f"Time since last movement: {time_since_last:.3f}s (frequency: {self.tracking_frequency:.3f}s)")
        
        if current_time - self.last_movement_time < self.tracking_frequency:
            print(f"SKIPPING: Too soon since last update")
            return
        
        if not faces:
            # Handle lost face tracking
            if self.last_known_face is not None:
                time_since_lost = current_time - self.face_lost_time if self.face_lost_time > 0 else 0
                
                if time_since_lost == 0:  # First time losing face
                    self.face_lost_time = current_time
                    print(f"FACE LOST: Starting timeout tracking ({self.face_lost_timeout}s)")
                
                if time_since_lost < self.face_lost_timeout:
                    print(f"USING LAST KNOWN FACE: {time_since_lost:.1f}s ago (timeout in {self.face_lost_timeout - time_since_lost:.1f}s)")
                    # Use the last known face position
                    faces = [self.last_known_face]
                else:
                    print(f"TIMEOUT REACHED: Stopping tracking after {self.face_lost_timeout}s")
                    self.last_known_face = None
                    self.face_lost_time = 0
                    self.last_tracking_position = None
                    return
            else:
                print(f"NO FACES: No current or last known face to track")
                return
        
        # Get the largest face and update last known position
        best_face = max(faces, key=lambda f: f['width'] * f['height'])
        print(f"Best face: center=({best_face['center_x']:.3f}, {best_face['center_y']:.3f}), size=({best_face['width']:.3f}x{best_face['height']:.3f})")
        
        # Determine if this is fresh face detection or cached last known face
        is_fresh_detection = self.face_lost_time == 0
        
        if is_fresh_detection:
            # This is a real fresh face detection
            self.last_known_face = best_face.copy()  # Store copy for future use
            print(f"FRESH FACE: Stored new last known position")
        else:
            # We're using cached last known face - check if we should recover
            if len(faces) > 0 and faces[0] != self.last_known_face:
                # New real detection after being lost
                self.last_known_face = best_face.copy()
                print(f"FACE RECOVERED: After {current_time - self.face_lost_time:.1f}s")
                self.face_lost_time = 0  # Reset lost timer
            else:
                print(f"TRACKING LOST FACE: {current_time - self.face_lost_time:.1f}s ago")
        
        # Calculate deviation from center
        center_x = best_face['center_x']
        center_y = best_face['center_y']
        face_size = best_face['width'] * best_face['height']
        
        deviation_x = center_x - 0.5
        deviation_y = center_y - 0.5
        
        print(f"\n--- DEADZONE CHECK ---")
        print(f"Deviation: x={deviation_x:.3f}, y={deviation_y:.3f}")
        print(f"Deadzone: {self.deadzone:.3f}")
        print(f"X outside deadzone: {abs(deviation_x) > self.deadzone}")
        print(f"Y outside deadzone: {abs(deviation_y) > self.deadzone}")
        
        # Apply deadzone - but make it much smaller for responsiveness
        if abs(deviation_x) < self.deadzone and abs(deviation_y) < self.deadzone:
            print(f"RESULT: Within deadzone - NO MOVEMENT")
            return
        else:
            print(f"RESULT: Outside deadzone - PROCEEDING WITH TRACKING")
        
        # Get current camera position and app state
        current_pan = self.camera_controller.current_pan_deg
        current_tilt = self.camera_controller.current_tilt_deg
        current_zoom = self.camera_controller.current_zoom_val
        mirror_enabled = getattr(self, 'mirror_enabled', False)
        
        print(f"\n--- CAMERA STATE ---")
        print(f"Current position: pan={current_pan:.1f}°, tilt={current_tilt:.1f}°, zoom={current_zoom}")
        print(f"Mirror enabled: {mirror_enabled}")
        
        # DEFINITIVE MOVEMENT CALCULATION - determine exact degrees needed
        
        # Calculate field of view compensation based on zoom
        # At zoom=0 (1x), camera has ~70° horizontal FOV
        # At zoom=996 (12x), camera has ~6° horizontal FOV
        base_fov_horizontal = 70  # degrees at 1x zoom
        base_fov_vertical = 40    # degrees at 1x zoom (roughly 16:9 aspect ratio)
        
        zoom_factor = 1.0 + (current_zoom / 996.0) * 11.0  # 1x to 12x
        current_fov_h = base_fov_horizontal / zoom_factor
        current_fov_v = base_fov_vertical / zoom_factor
        
        print(f"\n--- DEFINITIVE CALCULATION ---")
        print(f"Zoom: {current_zoom}/996 ({zoom_factor:.1f}x), FOV: {current_fov_h:.1f}°h x {current_fov_v:.1f}°v")
        
        # Calculate exact degrees needed to center the face
        # deviation_x/y is in frame coordinates (-0.5 to +0.5)
        # Convert to degrees of camera movement needed
        
        # If face is at deviation_x=0.5 (right edge), it needs to move left by half the FOV
        degrees_per_frame_width = current_fov_h
        degrees_per_frame_height = current_fov_v
        
        # Calculate exact movement needed with zoom-aware scaling
        pan_multiplier = 1 if mirror_enabled else -1  # Direction based on mirror
        
        # Scale movement based on zoom level to prevent overshooting
        # At wide zoom (1x), use smaller multiplier to prevent overshooting
        # At tight zoom (12x), use larger multiplier for precision
        zoom_scale_factor = 0.3 + (zoom_factor - 1.0) / 11.0 * 0.7  # 0.3 at 1x, 1.0 at 12x
        
        required_pan_move = deviation_x * degrees_per_frame_width * pan_multiplier * zoom_scale_factor
        required_tilt_move = deviation_y * degrees_per_frame_height * -1 * zoom_scale_factor  # Always opposite for centering
        
        print(f"Face deviation: ({deviation_x:.3f}, {deviation_y:.3f})")
        print(f"Zoom scale factor: {zoom_scale_factor:.2f} (prevents overshooting at wide zoom)")
        print(f"Raw movement: pan={deviation_x * degrees_per_frame_width * pan_multiplier:.2f}°, tilt={deviation_y * degrees_per_frame_height * -1:.2f}°")
        print(f"Scaled movement: pan={required_pan_move:.2f}°, tilt={required_tilt_move:.2f}°")
        
        # Apply movement only if it's significant enough (avoid micro-movements)
        min_movement_degrees = 1.0  # Only move if >= 1 degree needed
        
        new_pan = current_pan
        new_tilt = current_tilt
        
        if abs(required_pan_move) >= min_movement_degrees:
            new_pan = int(max(-169, min(169, current_pan + required_pan_move)))
        
        if abs(required_tilt_move) >= min_movement_degrees:
            new_tilt = int(max(-29, min(89, current_tilt + required_tilt_move)))
        
        print(f"Calculated target: pan={new_pan}°, tilt={new_tilt}°")
        print(f"Movement needed: pan={abs(new_pan - current_pan)}°, tilt={abs(new_tilt - current_tilt)}°")
        
        # Auto-zoom calculation based on face size and tightness setting
        if hasattr(self, 'auto_zoom_enabled') and self.auto_zoom_enabled:
            print(f"\n--- AUTO ZOOM CALCULATION ---")
            face_size = best_face['width'] * best_face['height']
            
            # Calculate ideal face size based on tightness setting
            # Tightness range: 0.1 (loose, 5% of frame) to 0.8 (tight, 25% of frame)
            tightness = getattr(self, 'tightness', 0.3)  # Default if not set
            print(f"DEBUG: Current tightness value: {tightness:.3f}")
            
            min_face_size = 0.01  # 1% at minimum tightness (much more zoomed out)
            max_face_size = 0.25  # 25% at maximum tightness
            ideal_face_size = min_face_size + (tightness * (max_face_size - min_face_size))
            
            print(f"DEBUG: Calculated ideal face size: {ideal_face_size:.3f} (from tightness {tightness:.3f})")
            
            size_ratio = face_size / ideal_face_size
            
            print(f"Face size: {face_size:.3f}, ideal: {ideal_face_size:.3f} (tightness: {tightness:.2f}), ratio: {size_ratio:.2f}")
            
            # Definitive zoom calculation with tighter tolerances
            if size_ratio < 0.8:  # Face too small - zoom in
                zoom_adjustment = min(60, (0.8 - size_ratio) * 200)  # More aggressive
                target_zoom = min(996, current_zoom + zoom_adjustment)
                print(f"ZOOM IN: Face too small ({size_ratio:.2f} < 0.8) → target ideal: {ideal_face_size:.3f}")
            elif size_ratio > 1.2:  # Face too big - zoom out
                zoom_adjustment = min(60, (size_ratio - 1.2) * 200)  # More aggressive
                target_zoom = max(0, current_zoom - zoom_adjustment)
                print(f"ZOOM OUT: Face too big ({size_ratio:.2f} > 1.2) → target ideal: {ideal_face_size:.3f}")
            else:
                target_zoom = current_zoom
                print(f"ZOOM OK: Face size is good ({size_ratio:.2f}) → ideal: {ideal_face_size:.3f}")
            
            # Apply definitive zoom if needed
            if abs(target_zoom - current_zoom) > 8:  # Lower threshold for more responsive zoom
                zoom_success = self.camera_controller.set_direct_zoom_value(int(target_zoom))
                status = "SUCCESS" if zoom_success else "FAILED"
                print(f"AUTO ZOOM: {current_zoom} → {int(target_zoom)} ({status}) [adjustment: {target_zoom - current_zoom:+.0f}]")
            else:
                print(f"NO ZOOM: Change too small ({abs(target_zoom - current_zoom):.1f} < 8)")
        else:
            print(f"Auto zoom: DISABLED (auto_zoom_enabled={getattr(self, 'auto_zoom_enabled', False)})")
        
        print(f"\n--- MOVEMENT EXECUTION ---")
        
        # Execute definitive movement only if position actually needs to change
        if new_pan != current_pan or new_tilt != current_tilt:
            print(f"EXECUTING DEFINITIVE MOVEMENT...")
            print(f"Moving from ({current_pan}°, {current_tilt}°) to ({new_pan}°, {new_tilt}°)")
            
            pan_success = True
            tilt_success = True
            
            if new_pan != current_pan:
                pan_success = self.camera_controller.set_direct_pan_degrees(new_pan)
                print(f"Pan movement: {current_pan}° → {new_pan}° ({'SUCCESS' if pan_success else 'FAILED'})")
            
            if new_tilt != current_tilt:
                tilt_success = self.camera_controller.set_direct_tilt_degrees(new_tilt)
                print(f"Tilt movement: {current_tilt}° → {new_tilt}° ({'SUCCESS' if tilt_success else 'FAILED'})")
            
            if pan_success and tilt_success:
                # Store the successful movement position for lost face tracking
                self.last_tracking_position = {
                    'pan': new_pan,
                    'tilt': new_tilt,
                    'face_center': (best_face['center_x'], best_face['center_y']),
                    'timestamp': current_time
                }
                status = "(using last known)" if self.face_lost_time > 0 else ""
                print(f"✓ MOVEMENT COMPLETED {status}: Face should be more centered now")
            else:
                print(f"✗ MOVEMENT FAILED: Camera may not have responded")
        else:
            if abs(required_pan_move) < min_movement_degrees and abs(required_tilt_move) < min_movement_degrees:
                status = "(last known face)" if self.face_lost_time > 0 else ""
                print(f"NO MOVEMENT {status}: Required movement too small (pan:{required_pan_move:.2f}°, tilt:{required_tilt_move:.2f}° < {min_movement_degrees}°)")
            else:
                status = "(last known face)" if self.face_lost_time > 0 else ""
                print(f"NO MOVEMENT {status}: Face is already well-centered")
        
        print(f"=== END TRACKING UPDATE ===\n")
        
        self.last_movement_time = current_time
        print(f"Updated last_movement_time to {current_time:.3f}")
    
    def set_auto_zoom_enabled(self, enabled):
        """Enable/disable auto zoom functionality"""
        self.auto_zoom_enabled = enabled
        print(f"TrackingController: Auto zoom set to {enabled}")
    
    def reset_lost_face_tracking(self):
        """Reset lost face tracking state - useful when manually moving camera"""
        self.last_known_face = None
        self.face_lost_time = 0
        self.last_tracking_position = None
        print("Lost face tracking state reset")
    


class VideoThread(QThread):
    """Video capture and processing thread"""
    
    frame_ready = pyqtSignal(np.ndarray)  # Frame with overlays for GUI preview
    clean_frame_ready = pyqtSignal(np.ndarray)  # Clean frame for virtual webcam
    faces_detected = pyqtSignal(list)
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.camera = VideoCapture()
        self.face_tracker = FaceTracker()
        self.running = False
        self.face_tracking_enabled = False
        self.face_detection_counter = 0
        self.last_faces = []
        self.face_detection_frequency = 4  # Default: every 4th frame
    
    def start_camera(self, camera_index):
        """Start camera capture"""
        if self.camera.start(camera_index):
            self.status_update.emit(f"Camera {camera_index} started")
            return True
        else:
            self.status_update.emit(f"Failed to start camera {camera_index}")
            return False
    
    def set_face_tracking(self, enabled):
        """Enable/disable face tracking"""
        self.face_tracking_enabled = enabled
    
    def set_face_detection_frequency(self, frequency):
        """Set how often face detection runs (every Nth frame)"""
        self.face_detection_frequency = max(1, frequency)  # Ensure at least every frame
        print(f"VideoThread: Face detection frequency set to every {self.face_detection_frequency}{'st' if frequency==1 else 'th'} frame")
    
    def run(self):
        """Main video processing loop - optimized for face tracking"""
        self.running = True
        
        while self.running and self.camera.is_opened():
            frame = self.camera.read_frame()
            if frame is not None:
                # Detect faces if tracking enabled - with frame skipping
                faces = []
                if self.face_tracking_enabled and MEDIAPIPE_AVAILABLE:
                    self.face_detection_counter += 1
                    
                    # Run face detection based on frequency setting
                    if self.face_detection_counter % self.face_detection_frequency == 0:
                        try:
                            faces = self.face_tracker.detect_faces(frame)
                            if faces:
                                self.last_faces = faces  # Cache successful detection
                                print(f"VideoThread: Detected {len(faces)} faces, emitting to app...")
                                self.faces_detected.emit(faces)
                            else:
                                print(f"VideoThread: No faces detected this frame")
                        except Exception as e:
                            print(f"Face detection failed: {e}")
                            faces = self.last_faces  # Use cached faces
                    else:
                        # Use cached faces for tracking continuity
                        faces = self.last_faces
                    
                # Emit clean frame first (before adding overlays) - only if needed for virtual webcam
                self.clean_frame_ready.emit(frame.copy())
                
                # Create preview frame with overlays
                preview_frame = frame.copy()
                
                if self.face_tracking_enabled:
                    # Draw face rectangles for preview only
                    for face in faces:
                        h, w = preview_frame.shape[:2]
                        x = int(face['x'] * w)
                        y = int(face['y'] * h)
                        face_w = int(face['width'] * w)
                        face_h = int(face['height'] * h)
                        
                        cv2.rectangle(preview_frame, (x, y), (x + face_w, y + face_h), (0, 255, 0), 2)
                        confidence_text = f"{face['confidence']:.2f}"
                        cv2.putText(preview_frame, confidence_text, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                    # Add tracking status overlay for preview only
                    status_text = f"FACE TRACKING: {'ON' if faces else 'SEARCHING'}"
                    color = (0, 255, 0) if faces else (0, 255, 255)
                else:
                    status_text = "MANUAL MODE"
                    color = (0, 0, 255)
                
                cv2.putText(preview_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                self.frame_ready.emit(preview_frame)
            
            self.msleep(33)  # ~30 FPS
    
    def stop(self):
        """Stop video processing"""
        self.running = False
        self.camera.stop()


class SettingsManager:
    """Save/load application settings"""
    
    def __init__(self, settings_file="cam_settings.json"):
        self.settings_file = settings_file
        self.default_settings = {
            "camera_ip": "localhost:36680",
            "last_camera_index": 0,
            "deadzone": 10,
            "tightness": 50,
            "virtual_webcam_enabled": False,
            # Camera settings defaults
            "mirror_enabled": True,
            "lowlight_enabled": False,
            "noise_reduction": "off",
            "sharpness": 0,  # 0 = off
            "white_balance_mode": "manual",
            "white_balance_value": 4000,
            "saturation": 4,
            "detection_frequency": 4,  # Every 4th frame for face detection
            "face_confidence": 50      # 50% confidence threshold
        }
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
        
        return self.default_settings.copy()
    
    def save_settings(self, settings):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            print("Settings saved")
        except Exception as e:
            print(f"Error saving settings: {e}")


class CameraControlApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.load_settings()
        
        self.camera_controller = HTTPCameraController(self.settings["camera_ip"])
        self.tracking_controller = TrackingController(self.camera_controller)
        self.video_thread = VideoThread()
        self.virtual_webcam = None
        
        self.setup_ui()
        self.connect_signals()
        self.load_settings_to_ui()
        
        # Load serial number
        self.load_serial_number()
        
        # Detect cameras
        self.detect_cameras()
        
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("AVer CAM520 Pro Controller - HTTP Edition")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(splitter)
        
        # Left panel - Video preview
        left_panel = self.create_video_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Controls
        right_panel = self.create_controls_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
    
    def create_video_panel(self):
        """Create video preview panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Video preview
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("No Camera Selected")
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setScaledContents(False)
        preview_layout.addWidget(self.preview_label)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
        preview_layout.addWidget(self.status_label)
        
        layout.addWidget(preview_group)
        
        # Connection settings
        connection_group = QGroupBox("Connection")
        connection_layout = QFormLayout(connection_group)
        
        # Camera IP input
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("localhost:36680")
        self.ip_input.textChanged.connect(self.on_ip_changed)
        connection_layout.addRow("Camera IP:", self.ip_input)
        
        # Serial number input
        self.serial_input = QLineEdit()
        self.serial_input.setPlaceholderText("5203561500051")
        self.serial_input.textChanged.connect(self.on_serial_changed)
        connection_layout.addRow("Serial Number:", self.serial_input)
        
        # Test connection button
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self.test_connection)
        connection_layout.addRow("", self.test_btn)
        
        # Camera selection
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selected)
        connection_layout.addRow("Video Source:", self.camera_combo)
        
        # Refresh cameras button
        self.refresh_btn = QPushButton("Refresh Cameras")
        self.refresh_btn.clicked.connect(self.detect_cameras)
        connection_layout.addRow("", self.refresh_btn)
        
        layout.addWidget(connection_group)
        
        # Virtual webcam
        virt_group = QGroupBox("Virtual Webcam")
        virt_layout = QVBoxLayout(virt_group)
        
        self.virtual_cam_btn = QPushButton("Start Virtual Webcam")
        self.virtual_cam_btn.clicked.connect(self.toggle_virtual_webcam)
        self.virtual_cam_btn.setEnabled(False)
        virt_layout.addWidget(self.virtual_cam_btn)
        
        self.virtual_status = QLabel("Not started")
        self.virtual_status.setStyleSheet("font-style: italic; color: gray;")
        virt_layout.addWidget(self.virtual_status)
        
        layout.addWidget(virt_group)
        
        return panel
    
    def create_controls_panel(self):
        """Create controls panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Face tracking
        tracking_group = QGroupBox("Face Tracking")
        tracking_layout = QVBoxLayout(tracking_group)
        
        self.tracking_checkbox = QCheckBox("Enable Face Tracking")
        self.tracking_checkbox.toggled.connect(self.toggle_face_tracking)
        tracking_layout.addWidget(self.tracking_checkbox)
        
        # Deadzone control
        deadzone_layout = QFormLayout()
        self.deadzone_slider = QSlider(Qt.Orientation.Horizontal)
        self.deadzone_slider.setRange(5, 50)  # Minimum 5% to prevent too much jitter
        self.deadzone_slider.valueChanged.connect(self.update_deadzone)
        self.deadzone_label = QLabel("20%")
        deadzone_row = QHBoxLayout()
        deadzone_row.addWidget(self.deadzone_slider)
        deadzone_row.addWidget(self.deadzone_label)
        
        # Add tooltip
        self.deadzone_slider.setToolTip("How close to center before camera moves\nHigher = less sensitive, less jittery")
        
        deadzone_layout.addRow("Movement Deadzone:", deadzone_row)
        
        # Tightness control (re-enabled with better logic)
        self.tightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.tightness_slider.setRange(10, 80)
        self.tightness_slider.valueChanged.connect(self.update_tightness)
        self.tightness_slider.setEnabled(False)  # Start disabled until face tracking enabled
        self.tightness_label = QLabel("30%")
        tightness_row = QHBoxLayout()
        tightness_row.addWidget(self.tightness_slider)
        tightness_row.addWidget(self.tightness_label)
        
        # Add tooltip
        self.tightness_slider.setToolTip("How tight to frame the face (zoom level)\nHigher = closer zoom on face")
        
        deadzone_layout.addRow("Auto-Zoom Level:", tightness_row)
        
        # Tracking frequency control
        self.frequency_slider = QSlider(Qt.Orientation.Horizontal)
        self.frequency_slider.setRange(2, 20)  # 2=0.02s (very fast), 20=2.0s (slow)
        self.frequency_slider.setValue(10)  # Default 0.1s for responsive tracking
        self.frequency_slider.valueChanged.connect(self.update_frequency)
        self.frequency_slider.setEnabled(False)  # Start disabled
        self.frequency_label = QLabel("0.1s")
        frequency_row = QHBoxLayout()
        frequency_row.addWidget(self.frequency_slider)
        frequency_row.addWidget(self.frequency_label)
        
        self.frequency_slider.setToolTip("How often camera adjusts\nVery Fast (0.02s) = ultra responsive\nFast (0.1s) = responsive tracking\nSlow (2.0s) = less frequent movements")
        
        deadzone_layout.addRow("Camera Movement Speed:", frequency_row)
        
        # Face detection frequency control
        self.detection_frequency_slider = QSlider(Qt.Orientation.Horizontal)
        self.detection_frequency_slider.setRange(1, 15)  # 1=every frame, 15=every 15th frame
        self.detection_frequency_slider.setValue(1)  # Default every frame (fastest detection)
        self.detection_frequency_slider.setMinimumWidth(150)  # Ensure it's wide enough to see
        self.detection_frequency_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.detection_frequency_slider.setTickInterval(2)  # Show ticks every 2 values
        self.detection_frequency_slider.valueChanged.connect(self.update_detection_frequency)
        self.detection_frequency_slider.setEnabled(False)  # Start disabled
        print(f"Created detection frequency slider with range 1-15, default value 4")
        self.detection_frequency_label = QLabel("Every 4th frame")
        self.detection_frequency_label.setMinimumWidth(100)  # Ensure label is visible
        detection_frequency_row = QHBoxLayout()
        detection_frequency_row.addWidget(self.detection_frequency_slider)
        detection_frequency_row.addWidget(self.detection_frequency_label)
        
        self.detection_frequency_slider.setToolTip("How often face detection runs\nEvery frame = most accurate but high CPU\nEvery 4th frame = good balance\nEvery 15th frame = low CPU but less responsive")
        
        deadzone_layout.addRow("Face Detection Speed:", detection_frequency_row)
        
        # Face detection confidence control
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(30, 90)  # 30% to 90% confidence
        self.confidence_slider.setValue(85)  # Default 85% confidence (fewer false positives)
        self.confidence_slider.setMinimumWidth(150)
        self.confidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.confidence_slider.setTickInterval(10)  # Show ticks every 10%
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.confidence_slider.setEnabled(False)  # Start disabled
        self.confidence_label = QLabel("50%")
        self.confidence_label.setMinimumWidth(50)
        confidence_row = QHBoxLayout()
        confidence_row.addWidget(self.confidence_slider)
        confidence_row.addWidget(self.confidence_label)
        
        self.confidence_slider.setToolTip("Face detection confidence threshold\nHigh (80%+) = fewer false positives, may miss some faces\nMedium (50%) = balanced accuracy\nLow (30%) = detects more faces but may pick up objects")
        
        deadzone_layout.addRow("Face Detection Confidence:", confidence_row)
        
        tracking_layout.addLayout(deadzone_layout)
        layout.addWidget(tracking_group)
        
        # PTZ Controls
        ptz_group = QGroupBox("Manual PTZ Controls")
        ptz_layout = QGridLayout(ptz_group)
        
        # PTZ buttons
        self.ptz_up_btn = QPushButton("▲ Up")
        self.ptz_down_btn = QPushButton("▼ Down")
        self.ptz_left_btn = QPushButton("◄ Left")
        self.ptz_right_btn = QPushButton("► Right")
        self.ptz_home_btn = QPushButton("⌂ Home")
        self.ptz_set_home_btn = QPushButton("Set Home")
        
        # Zoom buttons
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        
        # Focus buttons removed - functionality replaced by focus slider
        
        # Arrange buttons
        ptz_layout.addWidget(self.ptz_up_btn, 0, 1)
        ptz_layout.addWidget(self.ptz_left_btn, 1, 0)
        ptz_layout.addWidget(self.ptz_home_btn, 1, 1)
        ptz_layout.addWidget(self.ptz_right_btn, 1, 2)
        ptz_layout.addWidget(self.ptz_down_btn, 2, 1)
        ptz_layout.addWidget(self.ptz_set_home_btn, 0, 2)
        
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.zoom_out_btn)
        ptz_layout.addLayout(zoom_layout, 3, 0, 1, 3)
        
        # Focus layout removed - using focus slider instead
        
        self.connect_ptz_buttons()
        layout.addWidget(ptz_group)
        
        # Camera Settings (from sniffedtest.py)
        settings_group = QGroupBox("Camera Settings")
        settings_layout = QFormLayout(settings_group)
        
        # White balance - default manual as specified
        self.wb_combo = QComboBox()
        self.wb_combo.addItems(["Auto", "Manual"])
        self.wb_combo.setCurrentText("Manual")  # Default manual
        self.wb_combo.currentTextChanged.connect(self.update_wb_mode)
        settings_layout.addRow("White Balance:", self.wb_combo)
        
        self.wb_value_slider = QSlider(Qt.Orientation.Horizontal)
        self.wb_value_slider.setRange(2000, 6500)
        self.wb_value_slider.setValue(4000)
        self.wb_value_slider.setEnabled(False)
        self.wb_value_slider.valueChanged.connect(self.update_wb_value)
        self.wb_value_label = QLabel("4000K")
        wb_value_row = QHBoxLayout()
        wb_value_row.addWidget(self.wb_value_slider)
        wb_value_row.addWidget(self.wb_value_label)
        settings_layout.addRow("WB Value:", wb_value_row)
        
        # Saturation - default 4 as specified
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(1, 9)
        self.saturation_slider.setValue(4)  # Default to 4 instead of 5
        self.saturation_slider.valueChanged.connect(self.update_saturation)
        self.saturation_label = QLabel("4")  # Default to 4
        sat_row = QHBoxLayout()
        sat_row.addWidget(self.saturation_slider)
        sat_row.addWidget(self.saturation_label)
        settings_layout.addRow("Saturation:", sat_row)
        
        # Brightness (1-9)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(1, 9)
        self.brightness_slider.setValue(5)  # Default to middle value
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        self.brightness_label = QLabel("5")
        brightness_row = QHBoxLayout()
        brightness_row.addWidget(self.brightness_slider)
        brightness_row.addWidget(self.brightness_label)
        settings_layout.addRow("Brightness:", brightness_row)
        
        # Sharpness (corrected: dropdown with off/low/middle/high as specified)
        self.sharpness_combo = QComboBox()
        self.sharpness_combo.addItems(["Off", "Low", "Middle", "High"])
        self.sharpness_combo.setCurrentText("Off")
        self.sharpness_combo.currentTextChanged.connect(self.update_sharpness)
        settings_layout.addRow("Sharpness:", self.sharpness_combo)
        
        # Noise reduction - default off as specified
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["Off", "Low", "Middle", "High"])
        self.noise_combo.setCurrentText("Off")  # Default off
        self.noise_combo.currentTextChanged.connect(self.update_noise_reduction)
        settings_layout.addRow("Noise Reduction:", self.noise_combo)
        
        # Low light compensation - default off as specified
        self.lowlight_checkbox = QCheckBox("Low Light Compensation")
        self.lowlight_checkbox.setChecked(False)  # Default off
        self.lowlight_checkbox.toggled.connect(self.update_lowlight)
        settings_layout.addRow("", self.lowlight_checkbox)
        
        # Mirror - default enabled as specified
        self.mirror_checkbox = QCheckBox("Mirror/Flip")
        self.mirror_checkbox.setChecked(True)  # Default enabled
        self.mirror_checkbox.toggled.connect(self.update_mirror)
        settings_layout.addRow("", self.mirror_checkbox)
        
        # Focus controls
        self.focus_combo = QComboBox()
        self.focus_combo.addItems(["Auto", "Manual"])
        self.focus_combo.currentTextChanged.connect(self.update_focus_mode)
        settings_layout.addRow("Focus Mode:", self.focus_combo)
        
        self.focus_value_slider = QSlider(Qt.Orientation.Horizontal)
        self.focus_value_slider.setRange(0, 255)
        self.focus_value_slider.setValue(128)
        self.focus_value_slider.setEnabled(False)
        self.focus_value_slider.valueChanged.connect(self.update_focus_value)
        self.focus_value_label = QLabel("128")
        focus_value_row = QHBoxLayout()
        focus_value_row.addWidget(self.focus_value_slider)
        focus_value_row.addWidget(self.focus_value_label)
        settings_layout.addRow("Focus Value:", focus_value_row)
        
        layout.addWidget(settings_group)
        
        # Warning message
        warning_label = QLabel("⚠️ PTZApp 2 MUST be running for PTZ controls to work")
        warning_label.setStyleSheet("color: orange; font-weight: bold; padding: 10px; border: 1px solid orange; border-radius: 5px; background-color: rgba(255,165,0,0.1);")
        warning_label.setWordWrap(True)
        layout.addWidget(warning_label)
        
        layout.addStretch()
        return panel
    
    def connect_signals(self):
        """Connect signals from video thread"""
        self.video_thread.frame_ready.connect(self.update_preview)
        self.video_thread.clean_frame_ready.connect(self.send_to_virtual_webcam)
        self.video_thread.faces_detected.connect(self.handle_faces_detected)
        self.video_thread.status_update.connect(self.update_status)
    
    def connect_ptz_buttons(self):
        """Connect PTZ button events with continuous movement support"""
        # PTZ movement timers for continuous movement
        self.ptz_timers = {}
        
        # Directional movement - pressed/released for continuous movement
        self.ptz_up_btn.pressed.connect(lambda: self.start_continuous_movement('tilt', 2))     # Tilt up (positive)
        self.ptz_up_btn.released.connect(lambda: self.stop_continuous_movement('tilt'))
        
        self.ptz_down_btn.pressed.connect(lambda: self.start_continuous_movement('tilt', -2))  # Tilt down (negative) 
        self.ptz_down_btn.released.connect(lambda: self.stop_continuous_movement('tilt'))
        
        self.ptz_left_btn.pressed.connect(lambda: self.start_continuous_movement('pan', -2))   # Pan left
        self.ptz_left_btn.released.connect(lambda: self.stop_continuous_movement('pan'))
        
        self.ptz_right_btn.pressed.connect(lambda: self.start_continuous_movement('pan', 2))   # Pan right
        self.ptz_right_btn.released.connect(lambda: self.stop_continuous_movement('pan'))
        
        # Zoom using continuous movement
        self.zoom_in_btn.pressed.connect(lambda: self.start_continuous_movement('zoom', 15))   # Zoom in
        self.zoom_in_btn.released.connect(lambda: self.stop_continuous_movement('zoom'))
        
        self.zoom_out_btn.pressed.connect(lambda: self.start_continuous_movement('zoom', -15)) # Zoom out
        self.zoom_out_btn.released.connect(lambda: self.stop_continuous_movement('zoom'))
        
        # Focus buttons removed - using focus slider for direct control
        
        # Home buttons - update position tracking when going home
        self.ptz_home_btn.clicked.connect(self.go_home_and_update_position)
        self.ptz_set_home_btn.clicked.connect(self.camera_controller.set_home)
    
    def manual_pan_move(self, degrees):
        """Move camera pan by specified degrees"""
        # Apply mirror compensation
        mirror_enabled = getattr(self, 'mirror_enabled', False)
        if mirror_enabled:
            degrees = -degrees  # Reverse direction when mirrored
            
        current_pan = self.camera_controller.current_pan_deg
        new_pan = int(max(-169, min(169, current_pan + degrees)))
        if self.camera_controller.set_direct_pan_degrees(new_pan):
            print(f"Manual pan: {current_pan}° → {new_pan}°")
    
    def manual_tilt_move(self, degrees):
        """Move camera tilt by specified degrees"""
        current_tilt = self.camera_controller.current_tilt_deg
        new_tilt = int(max(-29, min(89, current_tilt + degrees)))
        if self.camera_controller.set_direct_tilt_degrees(new_tilt):
            print(f"Manual tilt: {current_tilt}° → {new_tilt}°")
    
    def manual_zoom_move(self, zoom_units):
        """Move camera zoom by specified units"""
        current_zoom = self.camera_controller.current_zoom_val
        new_zoom = max(0, min(996, current_zoom + zoom_units))
        if self.camera_controller.set_direct_zoom_value(new_zoom):
            print(f"Manual zoom: {current_zoom} → {new_zoom}")
    
    # manual_focus_move removed - focus controlled by slider instead
    
    def go_home_and_update_position(self):
        """Go home and reset position tracking"""
        if self.camera_controller.go_home():
            self.camera_controller.current_pan_deg = 0
            self.camera_controller.current_tilt_deg = 0
            self.camera_controller.current_zoom_val = 100
            print("Camera moved to home position (0°, 0°, zoom=100)")
    
    def start_continuous_movement(self, axis, step_size):
        """Start continuous movement in specified direction"""
        from PyQt6.QtCore import QTimer
        
        # Stop any existing timer for this axis
        self.stop_continuous_movement(axis)
        
        # Apply mirror compensation for pan movement
        if axis == 'pan':
            mirror_enabled = getattr(self, 'mirror_enabled', False)
            if mirror_enabled:
                step_size = -step_size  # Reverse direction when mirrored
        
        # Store movement parameters
        self.ptz_timers[axis] = {
            'timer': QTimer(),
            'step': step_size
        }
        
        # Connect timer to movement function
        if axis == 'pan':
            self.ptz_timers[axis]['timer'].timeout.connect(lambda: self.continuous_pan_step(step_size))
        elif axis == 'tilt':
            self.ptz_timers[axis]['timer'].timeout.connect(lambda: self.continuous_tilt_step(step_size))
        elif axis == 'zoom':
            self.ptz_timers[axis]['timer'].timeout.connect(lambda: self.continuous_zoom_step(step_size))
        
        # Start continuous movement (40ms intervals for smoother movement)
        self.ptz_timers[axis]['timer'].start(40)
        
        # Execute first step immediately
        if axis == 'pan':
            self.continuous_pan_step(step_size)
        elif axis == 'tilt':
            self.continuous_tilt_step(step_size)
        elif axis == 'zoom':
            self.continuous_zoom_step(step_size)
    
    def stop_continuous_movement(self, axis):
        """Stop continuous movement for specified axis"""
        if axis in self.ptz_timers and self.ptz_timers[axis]['timer'].isActive():
            self.ptz_timers[axis]['timer'].stop()
            print(f"Stopped {axis} movement")
    
    def continuous_pan_step(self, step_size):
        """Execute one pan step in continuous movement"""
        current_pan = self.camera_controller.current_pan_deg
        new_pan = int(max(-169, min(169, current_pan + step_size)))
        if new_pan != current_pan:  # Only move if within bounds
            self.camera_controller.set_direct_pan_degrees(new_pan)
            print(f"Continuous pan: {current_pan}° → {new_pan}°")
            # Reset lost face tracking since user manually moved camera
            self.tracking_controller.reset_lost_face_tracking()
    
    def continuous_tilt_step(self, step_size):
        """Execute one tilt step in continuous movement"""
        current_tilt = self.camera_controller.current_tilt_deg
        new_tilt = int(max(-29, min(89, current_tilt + step_size)))
        if new_tilt != current_tilt:  # Only move if within bounds
            self.camera_controller.set_direct_tilt_degrees(new_tilt)
            print(f"Continuous tilt: {current_tilt}° → {new_tilt}°")
            # Reset lost face tracking since user manually moved camera
            self.tracking_controller.reset_lost_face_tracking()
    
    def continuous_zoom_step(self, step_size):
        """Execute one zoom step in continuous movement"""
        current_zoom = self.camera_controller.current_zoom_val
        new_zoom = max(0, min(996, current_zoom + step_size))
        if new_zoom != current_zoom:  # Only move if within bounds
            self.camera_controller.set_direct_zoom_value(new_zoom)
            print(f"Continuous zoom: {current_zoom} → {new_zoom}")
    
    def load_settings_to_ui(self):
        """Load saved settings to UI"""
        self.ip_input.setText(self.settings["camera_ip"])
        self.deadzone_slider.setValue(self.settings.get("deadzone", 5))  # Default to 5% instead of 20%
        self.tightness_slider.setValue(self.settings["tightness"])
        # Trigger tightness update to enable auto zoom properly
        self.update_tightness(self.settings["tightness"])
        self.update_deadzone(self.settings.get("deadzone", 20))
        self.update_tightness(self.settings["tightness"])
        
        # Load frequency setting (default to 10 = 0.1s)
        freq_value = self.settings.get("frequency", 10)  # 10 = 0.1s with new conversion
        self.frequency_slider.setValue(freq_value)
        self.update_frequency(freq_value)
        
        # Load detection frequency setting (default to 4 = every 4th frame)
        detection_freq_value = self.settings.get("detection_frequency", 4)
        self.detection_frequency_slider.setValue(detection_freq_value)
        # Ensure the update method is called to set proper initial state
        self.update_detection_frequency(detection_freq_value)
        print(f"Detection frequency slider initialized to value {detection_freq_value}")
        
        # Load face detection confidence setting (default to 50%)
        confidence_value = self.settings.get("face_confidence", 50)
        self.confidence_slider.setValue(confidence_value)
        self.update_confidence(confidence_value)
        print(f"Face confidence slider initialized to {confidence_value}%")
        
        # Load and apply camera settings with defaults on startup
        self.apply_all_camera_settings()
    
    def apply_all_camera_settings(self):
        """Apply all camera settings with defaults on startup"""
        print("\n=== APPLYING ALL CAMERA SETTINGS ===")
        
        # Set UI controls to default values and apply settings
        
        # Mirror/Flip - default enabled
        mirror_enabled = self.settings.get("mirror_enabled", True)
        self.mirror_checkbox.setChecked(mirror_enabled)
        self.update_mirror(mirror_enabled)
        
        # Low Light - default off
        lowlight_enabled = self.settings.get("lowlight_enabled", False)
        self.lowlight_checkbox.setChecked(lowlight_enabled)
        self.update_lowlight(lowlight_enabled)
        
        # Noise Reduction - default off
        noise_level = self.settings.get("noise_reduction", "off")
        if hasattr(self, 'noise_combo'):
            index = self.noise_combo.findText(noise_level.title())
            if index >= 0:
                self.noise_combo.setCurrentIndex(index)
            self.update_noise_reduction(noise_level.title())
        
        # Sharpness - default off (0)
        sharpness_value = self.settings.get("sharpness", 0)
        if hasattr(self, 'sharpness_combo'):
            sharpness_names = ["Off", "Low", "Middle", "High"]
            if 0 <= sharpness_value < len(sharpness_names):
                self.sharpness_combo.setCurrentText(sharpness_names[sharpness_value])
            self.update_sharpness(sharpness_names[sharpness_value])
        
        # White Balance - default manual at 4000K
        wb_mode = self.settings.get("white_balance_mode", "manual")
        wb_value = self.settings.get("white_balance_value", 4000)
        if hasattr(self, 'wb_combo'):
            self.wb_combo.setCurrentText(wb_mode.title())
            self.wb_value_slider.setValue(wb_value)
            self.wb_value_label.setText(f"{wb_value}K")
            self.update_wb_mode(wb_mode.title())
            self.update_wb_value(wb_value)
        
        # Saturation - default 4
        saturation_value = self.settings.get("saturation", 4)
        if hasattr(self, 'saturation_slider'):
            self.saturation_slider.setValue(saturation_value)
            self.saturation_label.setText(str(saturation_value))
            self.update_saturation(saturation_value)
        
        # Brightness - default 5
        brightness_value = self.settings.get("brightness", 5)
        if hasattr(self, 'brightness_slider'):
            self.brightness_slider.setValue(brightness_value)
            self.brightness_label.setText(str(brightness_value))
            self.update_brightness(brightness_value)
        
        print("=== CAMERA SETTINGS APPLIED ===")
    
    def save_current_settings(self):
        """Save current settings"""
        self.settings["camera_ip"] = self.ip_input.text()
        self.settings["deadzone"] = self.deadzone_slider.value()
        self.settings["tightness"] = self.tightness_slider.value()
        self.settings["frequency"] = self.frequency_slider.value()
        self.settings["detection_frequency"] = self.detection_frequency_slider.value()
        self.settings["face_confidence"] = self.confidence_slider.value()
        if self.camera_combo.currentData() is not None:
            self.settings["last_camera_index"] = self.camera_combo.currentData()
        self.settings["virtual_webcam_enabled"] = self.virtual_webcam is not None and self.virtual_webcam.is_running()
        
        # Save current camera settings
        if hasattr(self, 'mirror_checkbox'):
            self.settings["mirror_enabled"] = self.mirror_checkbox.isChecked()
        if hasattr(self, 'lowlight_checkbox'):
            self.settings["lowlight_enabled"] = self.lowlight_checkbox.isChecked()
        if hasattr(self, 'noise_combo'):
            self.settings["noise_reduction"] = self.noise_combo.currentText().lower()
        if hasattr(self, 'sharpness_combo'):
            sharpness_map = {"Off": 0, "Low": 1, "Middle": 2, "High": 3}
            self.settings["sharpness"] = sharpness_map.get(self.sharpness_combo.currentText(), 0)
        if hasattr(self, 'wb_combo'):
            self.settings["white_balance_mode"] = self.wb_combo.currentText().lower()
        if hasattr(self, 'wb_value_slider'):
            self.settings["white_balance_value"] = self.wb_value_slider.value()
        if hasattr(self, 'saturation_slider'):
            self.settings["saturation"] = self.saturation_slider.value()
        if hasattr(self, 'brightness_slider'):
            self.settings["brightness"] = self.brightness_slider.value()
        
        self.settings_manager.save_settings(self.settings)
    
    def detect_cameras(self):
        """Detect available cameras"""
        self.camera_combo.clear()
        self.update_status("Detecting cameras...")
        
        try:
            cameras = get_all_cameras()
            
            if not cameras:
                self.camera_combo.addItem("No cameras found", None)
                self.update_status("No cameras detected")
            else:
                for camera in cameras:
                    label = f"{camera['name']} ({camera['resolution']})"
                    self.camera_combo.addItem(label, camera['index'])
                
                self.update_status(f"Found {len(cameras)} camera(s)")
                
                # Select last used camera if available
                last_index = self.settings.get("last_camera_index", 0)
                for i in range(self.camera_combo.count()):
                    if self.camera_combo.itemData(i) == last_index:
                        self.camera_combo.setCurrentIndex(i)
                        break
        
        except Exception as e:
            self.camera_combo.addItem("Detection failed", None)
            self.update_status(f"Camera detection failed: {e}")
    
    def on_ip_changed(self):
        """Handle IP address change"""
        ip = self.ip_input.text()
        self.camera_controller.set_camera_ip(ip)
        self.save_current_settings()
    
    def on_serial_changed(self):
        """Handle serial number change"""
        serial = self.serial_input.text()
        self.camera_controller.set_serial_number(serial)
        self.save_serial_number(serial)
    
    def load_serial_number(self):
        """Load serial number from file"""
        try:
            with open('serial_number.txt', 'r') as f:
                serial = f.read().strip()
                if serial:
                    self.serial_input.setText(serial)
                    self.camera_controller.set_serial_number(serial)
                    return serial
        except FileNotFoundError:
            pass
        return "5203561500051"  # Default serial number
    
    def save_serial_number(self, serial):
        """Save serial number to file"""
        try:
            with open('serial_number.txt', 'w') as f:
                f.write(serial)
        except Exception as e:
            print(f"Error saving serial number: {e}")
    
    def test_connection(self):
        """Test camera HTTP connection"""
        self.update_status("Testing connection...")
        
        if self.camera_controller.test_connection():
            self.update_status("✓ Camera connection successful")
            QMessageBox.information(self, "Connection Test", "Camera connection successful!")
        else:
            self.update_status("✗ Camera connection failed")
            QMessageBox.warning(self, "Connection Test", 
                               f"Failed to connect to camera at {self.camera_controller.camera_ip}\n\n"
                               "Please check:\n"
                               "• Camera IP address is correct\n"
                               "• Camera software is running\n"
                               "• Network connection")
    
    def on_camera_selected(self):
        """Handle camera selection"""
        camera_index = self.camera_combo.currentData()
        if camera_index is not None:
            # Save the selected camera immediately
            self.settings["last_camera_index"] = camera_index
            self.settings_manager.save_settings(self.settings)
            print(f"Saved camera index: {camera_index}")
            
            self.update_status("Starting camera...")
            
            if self.video_thread.start_camera(camera_index):
                if not self.video_thread.isRunning():
                    self.video_thread.start()
                self.virtual_cam_btn.setEnabled(True)
            else:
                self.virtual_cam_btn.setEnabled(False)
    
    def toggle_virtual_webcam(self):
        """Toggle virtual webcam on/off"""
        if self.virtual_webcam and self.virtual_webcam.is_running():
            # Stop virtual webcam
            self.virtual_webcam.stop()
            self.virtual_webcam = None
            self.virtual_cam_btn.setText("Start Virtual Webcam")
            self.virtual_status.setText("Stopped")
            self.virtual_status.setStyleSheet("color: gray;")
            self.update_status("Virtual webcam stopped")
        else:
            # Start virtual webcam
            self.virtual_webcam = VirtualWebcam(1920, 1080, 30)
            success, message = self.virtual_webcam.start()
            
            if success:
                self.virtual_cam_btn.setText("Stop Virtual Webcam")
                self.virtual_status.setText(f"Running: {message}")
                self.virtual_status.setStyleSheet("color: green;")
                self.update_status(f"Virtual webcam started: {message}")
                print("Virtual webcam is now available as 'OBS Virtual Camera' in other applications!")
            else:
                self.virtual_webcam = None
                self.virtual_status.setText(f"Failed: {message}")
                self.virtual_status.setStyleSheet("color: red;")
                self.update_status(f"Virtual webcam failed: {message}")
                
                # Show installation help if needed
                if "install" in message.lower() or "not found" in message.lower():
                    help_msg = """
Virtual Camera Setup Required:

To create a virtual webcam visible to other applications, you need:

1. Install OBS Studio (recommended):
   - Download from: https://obsproject.com/
   - Includes virtual camera filter
   - Works with all apps (Zoom, Teams, Discord, etc.)

2. Alternative: Install standalone OBS Virtual Camera filter

After installation, restart this application and the virtual webcam 
will appear as "OBS Virtual Camera" in other applications.
                    """
                    print(help_msg)
        
        self.save_current_settings()
    
    def toggle_face_tracking(self, enabled):
        """Toggle face tracking"""
        print(f"\n=== TOGGLING FACE TRACKING: {enabled} ===")
        
        # ALWAYS stop all movement first
        print("STOPPING ALL CAMERA MOVEMENT...")
        self.camera_controller.stop_all_movement()
        
        # Stop any ongoing smooth tracking and reset lost face tracking
        if hasattr(self, 'tracking_controller'):
            self.tracking_controller.is_smoothing = False
            # Reset lost face tracking when disabling
            if not enabled:
                self.tracking_controller.last_known_face = None
                self.tracking_controller.face_lost_time = 0
                self.tracking_controller.last_tracking_position = None
                print("Reset lost face tracking state")
        
        self.video_thread.set_face_tracking(enabled)
        
        # Enable/disable manual controls
        controls = [self.ptz_up_btn, self.ptz_down_btn, self.ptz_left_btn, 
                   self.ptz_right_btn, self.zoom_in_btn, self.zoom_out_btn]
        
        for control in controls:
            control.setEnabled(not enabled)
        
        # Enable/disable face tracking sliders and setup auto zoom
        print(f"Setting slider states to: {enabled}")
        self.deadzone_slider.setEnabled(enabled)
        self.tightness_slider.setEnabled(enabled)
        self.frequency_slider.setEnabled(enabled)
        self.detection_frequency_slider.setEnabled(enabled)
        self.confidence_slider.setEnabled(enabled)
        
        # Setup auto zoom based on current tightness
        if enabled:
            current_tightness = self.tightness_slider.value() / 100.0
            auto_zoom_enabled = current_tightness > 0.1
            self.tracking_controller.set_auto_zoom_enabled(auto_zoom_enabled)
            print(f"Tracking enabled → Auto zoom: {auto_zoom_enabled} (tightness: {current_tightness:.2f})")
        else:
            self.tracking_controller.set_auto_zoom_enabled(False)
            print(f"Tracking disabled → Auto zoom: False")
        
        status = "enabled" if enabled else "disabled"
        self.update_status(f"Face tracking {status}")
        print(f"=== FACE TRACKING {status.upper()} ===")
    
    def update_deadzone(self, value):
        """Update tracking deadzone"""
        self.deadzone_label.setText(f"{value}%")
        self.tracking_controller.set_deadzone(value / 100.0)
    
    def update_tightness(self, value):
        """Update tracking tightness (controls auto-zoom)"""
        self.tightness_label.setText(f"{value}%")
        tightness = value / 100.0
        self.tracking_controller.set_tightness(tightness)
        
        # Auto zoom is controlled by tightness - enable when > 10%
        auto_zoom_enabled = tightness > 0.1
        self.tracking_controller.set_auto_zoom_enabled(auto_zoom_enabled)
        
        # Calculate what the ideal face size will be
        min_face_size = 0.01  # 1% (much more zoomed out)
        max_face_size = 0.25  # 25%
        ideal_face_size = min_face_size + (tightness * (max_face_size - min_face_size))
        print(f"UI: Tightness {value}% ({tightness:.3f}) → Auto-zoom: {auto_zoom_enabled}, Target: {ideal_face_size*100:.1f}% of frame")
        
        # Debug: Check if tracking controller received the tightness
        if hasattr(self.tracking_controller, 'tightness'):
            print(f"UI: Tracking controller tightness is now: {self.tracking_controller.tightness:.3f}")
        else:
            print(f"UI: WARNING - Tracking controller has no tightness attribute!")
    
    def update_frequency(self, value):
        """Update camera movement frequency"""
        # New conversion: 2-20 to 0.02-2.0 seconds with better precision
        if value < 10:
            frequency_seconds = value / 100.0  # 2-9 → 0.02s-0.09s (ultra fast)
        else:
            frequency_seconds = (value - 10) / 10.0 + 0.1  # 10-20 → 0.1s-2.0s (normal)
        
        self.frequency_label.setText(f"{frequency_seconds:.2f}s")
        self.tracking_controller.set_frequency(frequency_seconds)
        print(f"Camera movement frequency: {frequency_seconds:.3f}s")
    
    def update_detection_frequency(self, value):
        """Update face detection frequency"""
        print(f"\nUpdate detection frequency called with value: {value}")
        
        if value == 1:
            label_text = "Every frame"
        elif value == 2:
            label_text = "Every 2nd frame"
        elif value == 3:
            label_text = "Every 3rd frame"
        else:
            label_text = f"Every {value}th frame"
        
        self.detection_frequency_label.setText(label_text)
        print(f"Detection frequency label set to: {label_text}")
        
        # Update the video thread's detection frequency
        if hasattr(self, 'video_thread'):
            self.video_thread.set_face_detection_frequency(value)
            print(f"Video thread detection frequency updated")
        else:
            print(f"Video thread not available yet")
        
        print(f"Face detection frequency: {label_text}\n")
    
    def update_confidence(self, value):
        """Update face detection confidence threshold"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{value}%")
        
        # Update the face tracker's confidence threshold
        if hasattr(self, 'video_thread') and self.video_thread.face_tracker:
            self.video_thread.face_tracker.set_confidence_threshold(confidence)
            print(f"Face detection confidence set to {value}% ({confidence:.2f})")
        else:
            print(f"Face tracker not available yet - confidence will be set on initialization")
    
    def handle_faces_detected(self, faces):
        """Handle detected faces - non-blocking"""
        tracking_checked = self.tracking_checkbox.isChecked()
        print(f"\nCameraControlApp.handle_faces_detected: {len(faces) if faces else 0} faces, checkbox_checked: {tracking_checked}")
        
        if tracking_checked and faces:
            print(f"Scheduling tracking update with {len(faces)} faces via QTimer...")
            # Use QTimer to defer camera control to prevent blocking video thread
            QTimer.singleShot(0, lambda: self.tracking_controller.update_tracking(faces))
        else:
            if not tracking_checked:
                print(f"Face tracking checkbox not checked - ignoring faces")
            if not faces:
                print(f"No faces to track")
    
    def update_preview(self, frame):
        """Update video preview with overlays"""
        # Resize for preview
        preview_frame = cv2.resize(frame, (640, 360))
        
        # Convert to Qt format
        rgb_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        self.preview_label.setPixmap(pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
    
    def send_to_virtual_webcam(self, clean_frame):
        """Send clean frame (without overlays) to virtual webcam"""
        if self.virtual_webcam and self.virtual_webcam.is_running():
            self.virtual_webcam.send_frame(clean_frame)
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
        print(f"Status: {message}")
    
    # Camera settings handlers (from sniffedtest.py)
    def update_wb_mode(self, mode):
        """Update white balance mode"""
        if mode == "Auto":
            self.camera_controller.set_white_balance("auto")
            self.wb_value_slider.setEnabled(False)
        else:
            self.camera_controller.set_white_balance("manual")
            self.wb_value_slider.setEnabled(True)
    
    def update_wb_value(self, value):
        """Update white balance value"""
        self.wb_value_label.setText(f"{value}K")
        if self.wb_combo.currentText() == "Manual":
            self.camera_controller.set_numeric_setting("wb", value)
    
    def update_saturation(self, value):
        """Update saturation (1-9 range)"""
        self.saturation_label.setText(str(value))
        success = self.camera_controller.set_saturation(value)
        print(f"Saturation set to {value}: {'SUCCESS' if success else 'FAILED'}")
    
    def update_brightness(self, value):
        """Update brightness (1-9 range)"""
        self.brightness_label.setText(str(value))
        success = self.camera_controller.set_brightness(value)
        print(f"Brightness set to {value}: {'SUCCESS' if success else 'FAILED'}")
    
    def update_sharpness(self, level):
        """Update sharpness level using dropdown"""
        level_map = {"Off": 0, "Low": 1, "Middle": 2, "High": 3}
        value = level_map.get(level, 2)
        success = self.camera_controller.set_sharpness(value)
        print(f"Sharpness set to {level} (value: {value}): {'SUCCESS' if success else 'FAILED'}")
    
    def update_noise_reduction(self, level):
        """Update noise reduction"""
        self.camera_controller.set_noise_reduction(level.lower())
    
    def update_lowlight(self, enabled):
        """Update low light compensation"""
        self.camera_controller.set_low_light(enabled)
    
    def update_mirror(self, enabled):
        """Update mirror/flip and track state"""
        self.mirror_enabled = enabled  # Track mirror state
        self.camera_controller.set_mirror(enabled)
        print(f"Mirror/flip: {enabled}")
    

    def update_focus_mode(self, mode):
        """Update focus mode using new commands"""
        if mode == "Auto":
            success = self.camera_controller.set_focus_mode("auto")
            self.focus_value_slider.setEnabled(False)
            print(f"Focus set to auto: {'SUCCESS' if success else 'FAILED'}")
        else:
            success = self.camera_controller.set_focus_mode("manual")
            self.focus_value_slider.setEnabled(True)
            print(f"Focus set to manual: {'SUCCESS' if success else 'FAILED'}")

    def update_focus_value(self, value):
        """Update focus value using new command"""
        self.focus_value_label.setText(str(value))
        if self.focus_combo.currentText() == "Manual":
            success = self.camera_controller.set_focus_value(value)
            print(f"Focus value set to {value}: {'SUCCESS' if success else 'FAILED'}")
    def closeEvent(self, event):
        """Handle application close"""
        self.save_current_settings()
        
        # Stop all movement
        self.camera_controller.stop_all_movement()
        
        # Stop video
        self.video_thread.stop()
        self.video_thread.wait()
        
        # Stop virtual webcam
        if self.virtual_webcam:
            self.virtual_webcam.stop()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = CameraControlApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
