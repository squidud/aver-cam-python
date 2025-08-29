#!/usr/bin/env python3
"""
Proper Virtual Webcam using OBS Virtual Camera filter (standalone)
"""
import numpy as np
import cv2
import threading
import time
import os
import subprocess
import sys
import tempfile
import zipfile
import requests

try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False


class VirtualWebcam:
    """Virtual webcam that creates a real camera device visible to all applications"""
    
    def __init__(self, width=1920, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.thread = None
        self.camera = None
        
        # Check if we have the virtual camera filter
        self.has_virtual_camera = self._check_virtual_camera_available()
    
    def _check_virtual_camera_available(self):
        """Check if OBS Virtual Camera filter is available"""
        if not PYVIRTUALCAM_AVAILABLE:
            return False
        
        try:
            # Try to create a test camera
            test_cam = pyvirtualcam.Camera(width=640, height=480, fps=30)
            device_name = test_cam.device
            test_cam.close()
            
            # Check if it's the dummy fallback or real device
            if "obs" in device_name.lower() or "virtual" in device_name.lower():
                return True
            return False
        except Exception as e:
            print(f"Virtual camera test failed: {e}")
            return False
    
    def _download_obs_virtual_camera(self):
        """Download and install OBS Virtual Camera filter standalone"""
        print("Attempting to install OBS Virtual Camera filter...")
        
        try:
            # Check if already installed by looking for registry entries or files
            import winreg
            
            # Check common installation locations
            possible_paths = [
                r"C:\Program Files\obs-studio\bin\64bit\obs-virtualcam-module.dll",
                r"C:\Program Files (x86)\obs-studio\bin\32bit\obs-virtualcam-module.dll",
                r"C:\Windows\System32\obs-virtualcam.dll",
                r"C:\Windows\SysWOW64\obs-virtualcam.dll"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found existing OBS Virtual Camera at: {path}")
                    return True
            
            print("OBS Virtual Camera not found. You can install it by:")
            print("1. Installing OBS Studio (includes virtual camera)")
            print("2. Or installing the standalone OBS Virtual Camera filter")
            print("3. Or using alternative virtual camera software")
            
            return False
            
        except Exception as e:
            print(f"Error checking for OBS Virtual Camera: {e}")
            return False
    
    def start(self):
        """Start virtual webcam"""
        try:
            print(f"Creating virtual webcam {self.width}x{self.height} @ {self.fps}fps...")
            
            # Check if pyvirtualcam is available
            if not PYVIRTUALCAM_AVAILABLE:
                return False, "pyvirtualcam not available. Install with: pip install pyvirtualcam"
            
            try:
                print("Creating pyvirtualcam camera...")
                self.camera = pyvirtualcam.Camera(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    fmt=pyvirtualcam.PixelFormat.RGB
                )
                
                device_name = self.camera.device
                print(f"SUCCESS: Virtual webcam created: {device_name}")
                
                # Test if we can send a frame
                test_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                test_frame[:, :] = [64, 128, 192]  # Test color
                self.camera.send(test_frame)
                print("SUCCESS: Test frame sent successfully")
                
            except Exception as e:
                print(f"Camera creation failed: {e}")
                if self.camera:
                    try:
                        self.camera.close()
                    except:
                        pass
                    self.camera = None
                
                # If it failed, provide helpful error message
                error_msg = f"Virtual camera failed: {str(e)}"
                if "obs" in str(e).lower() or "backend" in str(e).lower():
                    error_msg += "\n\nThe OBS Virtual Camera filter might not be properly installed."
                    error_msg += "\nTry: 1) Restart OBS Studio if running, 2) Reinstall OBS Studio"
                return False, error_msg
            
            if not self.camera:
                return False, "Failed to create virtual camera device"
            
            # Initialize with a test pattern
            test_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Create a test pattern
            cv2.putText(test_frame, 'CamControl Virtual Camera', 
                       (50, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       2, (255, 255, 255), 3)
            cv2.putText(test_frame, 'Waiting for video feed...', 
                       (50, self.height//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (128, 128, 128), 2)
            
            # Convert BGR to RGB for pyvirtualcam
            test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            
            with self.frame_lock:
                self.current_frame = test_frame_rgb
            
            # Start streaming thread
            self.running = True
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            
            device_name = self.camera.device
            success_msg = f"Virtual Camera: {device_name}"
            print(f"SUCCESS: {success_msg}")
            print("The virtual camera should now be visible in other applications!")
            
            return True, success_msg
            
        except Exception as e:
            print(f"✗ Failed to create virtual webcam: {e}")
            error_msg = f"Virtual camera failed: {str(e)}"
            
            # Provide helpful error message
            if "obs" in str(e).lower():
                error_msg += "\nTip: Install OBS Studio or OBS Virtual Camera filter"
            
            return False, error_msg
    
    def stop(self):
        """Stop virtual webcam"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
            self.camera = None
        
        print("Virtual webcam stopped")
    
    def send_frame(self, frame):
        """Send frame to virtual webcam"""
        if not self.running or not self.camera:
            return
        
        try:
            # Resize frame if needed
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Convert BGR to RGB for pyvirtualcam
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with self.frame_lock:
                self.current_frame = frame_rgb.copy()
                
        except Exception as e:
            print(f"Error processing frame for virtual webcam: {e}")
    
    def _stream_loop(self):
        """Main streaming loop"""
        frame_duration = 1.0 / self.fps
        
        while self.running and self.camera:
            start_time = time.time()
            
            # Get current frame
            with self.frame_lock:
                if self.current_frame is not None:
                    frame_to_send = self.current_frame.copy()
                else:
                    # Default test pattern
                    frame_to_send = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    cv2.putText(frame_to_send, 'No Signal', 
                               (self.width//2-100, self.height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    frame_to_send = cv2.cvtColor(frame_to_send, cv2.COLOR_BGR2RGB)
            
            # Send frame to virtual camera
            try:
                self.camera.send(frame_to_send)
            except Exception as e:
                print(f"Error sending frame to virtual webcam: {e}")
                break
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def is_running(self):
        """Check if virtual webcam is running"""
        return self.running and self.camera is not None
    
    def get_installation_help(self):
        """Get help text for installing virtual camera"""
        help_text = """
To use the virtual webcam, you need OBS Virtual Camera:

Option 1 - Install OBS Studio (Recommended):
1. Download OBS Studio from: https://obsproject.com/
2. Install it (includes virtual camera filter)
3. Restart this application

Option 2 - Standalone Virtual Camera Filter:
1. Download OBS Virtual Camera from GitHub releases
2. Install the filter manually
3. Restart this application

Option 3 - Alternative Virtual Camera Software:
- ManyCam
- XSplit VCam  
- NVIDIA Broadcast

After installation, the virtual webcam will appear as a camera device 
in other applications like Zoom, Teams, Discord, etc.
        """
        return help_text.strip()