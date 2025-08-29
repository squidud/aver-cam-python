#!/usr/bin/env python3
"""
Simplified camera detection and video handling
"""
import cv2


def get_all_cameras():
    """Get list of all available cameras with their actual device names"""
    cameras = []
    
    print("Detecting cameras...")
    
    # Try to get device names using cv2-enumerate-cameras if available
    device_names = {}
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        print("Using cv2-enumerate-cameras for device names...")
        
        for camera_info in enumerate_cameras():
            # Map device names to indices (best effort)
            device_names[len(device_names)] = camera_info.name
            print(f"Found device: {camera_info.name}")
            
    except ImportError:
        print("cv2-enumerate-cameras not available, using basic names")
    
    # Test camera indices 0-9 and get actual working cameras
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                # Get basic info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Use actual device name if available, otherwise generic name
                if i in device_names:
                    device_name = device_names[i]
                else:
                    device_name = f"Camera {i}"
                
                cameras.append({
                    'index': i,
                    'name': device_name,
                    'resolution': f"{width}x{height}",
                    'working': True
                })
                print(f"Found Camera {i}: {device_name} ({width}x{height})")
            cap.release()
    
    print(f"Total cameras found: {len(cameras)}")
    return cameras


class VideoCapture:
    """Simple video capture wrapper"""
    
    def __init__(self):
        self.cap = None
        self.camera_index = None
    
    def start(self, camera_index):
        """Start capturing from camera"""
        if self.cap:
            self.cap.release()
        
        print(f"Starting camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.camera_index = camera_index
        
        if self.cap.isOpened():
            # Set reasonable defaults
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify settings
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            print(f"Camera {camera_index} started: {width}x{height} @ {fps}fps")
            
            return True
        return False
    
    def read_frame(self):
        """Read a frame from camera"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def stop(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
            print(f"Camera {self.camera_index} stopped")
    
    def is_opened(self):
        """Check if camera is opened"""
        return self.cap and self.cap.isOpened()