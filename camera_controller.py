#!/usr/bin/env python3
"""
HTTP-based camera controller using the method from sniffedtest.py
"""
import requests
import time
import threading


class HTTPCameraController:
    """Controls AVer CAM520 Pro via HTTP requests (based on sniffedtest.py)"""
    
    def __init__(self, camera_ip="localhost:36680"):
        self.camera_ip = camera_ip
        self.base_url = f"http://{camera_ip}"
        self.serial_number = "5203561500051"  # Default serial number
        self.active_movements = set()  # Track active movements
        self.movement_lock = threading.Lock()
        
        # Position tracking for fluid movement
        self.current_pan_deg = 0
        self.current_tilt_deg = 0
        self.current_zoom_val = 100
        
        # PTZ movement axes from sniffedtest.py
        self.AXES = {
            "left": ("ptz?action=left1", "ptz?action=left0"),
            "right": ("ptz?action=right1", "ptz?action=right0"),
            "up": ("ptz?action=up1", "ptz?action=up0"),
            "down": ("ptz?action=down1", "ptz?action=down0"),
            "zoomin": ("ptz?action=zoomin1", "ptz?action=zoomin0"),
            "zoomout": ("ptz?action=zoomout1", "ptz?action=zoomout0"),
            "focusin": ("ptz?action=focusin1", "ptz?action=focusin0"),
            "focusout": ("ptz?action=focusout1", "ptz?action=focusout0"),
        }
        
        # Camera settings from sniffedtest.py - will be dynamically updated with serial number
        self.SETTINGS = {
            "lowlight_on": [
                "setting?action=googleanalyticsevent&eventcategory=camera&eventaction=low%20light%20compensation%20change&eventlabel=light%3AOn",
                "setting?action=setcmd&cmdtype=UVC&selector=14&value=1&UVCID={serial}"
            ],
            "lowlight_off": [
                "setting?action=googleanalyticsevent&eventcategory=camera&eventaction=low%20light%20compensation%20change&eventlabel=light%3AOff",
                "setting?action=setcmd&cmdtype=UVC&selector=14&value=0&UVCID={serial}"
            ],
            "noise_reduction_off": [
                "setting?action=googleanalyticsevent&eventcategory=camera&eventaction=image%20noise%20reduction%20change&eventlabel=noise%3AOff",
                "setting?action=setcmd&cmdtype=UVCX1&selector=14&value=0&UVCID={serial}"
            ],
            "noise_reduction_low": [
                "setting?action=setcmd&cmdtype=UVCX1&selector=14&value=1&UVCID={serial}",
                "setting?action=googleanalyticsevent&eventcategory=camera&eventaction=image%20noise%20reduction%20change&eventlabel=noise%3ALow"
            ],
            "noise_reduction_middle": [
                "setting?action=googleanalyticsevent&eventcategory=camera&eventaction=image%20noise%20reduction%20change&eventlabel=noise%3AMiddle",
                "setting?action=setcmd&cmdtype=UVCX1&selector=14&value=2&UVCID={serial}"
            ],
            "noise_reduction_high": [
                "setting?action=googleanalyticsevent&eventcategory=camera&eventaction=image%20noise%20reduction%20change&eventlabel=noise%3AHigh",
                "setting?action=setcmd&cmdtype=UVCX1&selector=14&value=3&UVCID={serial}"
            ],
            "wb_manual": [
                "setting?action=getcmd&cmdtype=UVC&selector=5&UVCID={serial}",
                "setting?action=setcmd&cmdtype=UVC&selector=6&value=0&UVCID={serial}"
            ],
            "wb_auto": [
                "setting?action=getcmd&cmdtype=UVC&selector=5&UVCID={serial}",
                "setting?action=setcmd&cmdtype=UVC&selector=6&value=1&UVCID={serial}"
            ],
            "mirror_true": ["setting?action=setcmd&cmdtype=UVCX1&selector=16&value=1&UVCID={serial}"],
            "mirror_false": ["setting?action=setcmd&cmdtype=UVCX1&selector=16&value=0&UVCID={serial}"],
            "focus_auto": ["setting?action=setcmd&cmdtype=UVC&selector=8&value=1&UVCID={serial}"],
            "focus_manual": ["setting?action=setcmd&cmdtype=UVC&selector=8&value=0&UVCID={serial}"],
            "home_go": ["ptz?action=gopreset&index=0"],
            "home_set": ["ptz?action=setpreset&index=0"],
        }
        
        # Numeric settings from sniffedtest.py
        self.NUMERIC_SETTINGS = {
            "wb": 5,           # white balance value (manual mode)
            "saturation": 3,
            "sharpness": 2,
            "focus": 7         # focus value (manual mode)
        }
    
    def set_camera_ip(self, camera_ip):
        """Update the camera IP address"""
        self.camera_ip = camera_ip
        self.base_url = f"http://{camera_ip}"
        print(f"Camera IP updated to: {camera_ip}")
    
    def set_serial_number(self, serial_number):
        """Update the camera serial number"""
        self.serial_number = serial_number
        print(f"Camera serial number updated to: {serial_number}")
    
    def send_command(self, path):
        """Send HTTP command to camera"""
        url = f"{self.base_url}/{path}"
        try:
            response = requests.get(url, timeout=2)
            print(f"Sent {path}: {response.status_code}")
            return response.status_code == 200
        except requests.RequestException as e:
            print(f"Error sending {path}: {e}")
            return False
    
    def test_connection(self):
        """Test if camera is responding"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def start_movement(self, *directions):
        """Start PTZ movement in given directions"""
        with self.movement_lock:
            for direction in directions:
                if direction in self.AXES and direction not in self.active_movements:
                    start_cmd = self.AXES[direction][0]
                    if self.send_command(start_cmd):
                        self.active_movements.add(direction)
                        print(f"Started {direction} movement")
    
    def stop_movement(self, *directions):
        """Stop PTZ movement in given directions"""
        with self.movement_lock:
            for direction in directions:
                if direction in self.AXES and direction in self.active_movements:
                    stop_cmd = self.AXES[direction][1]
                    if self.send_command(stop_cmd):
                        self.active_movements.discard(direction)
                        print(f"Stopped {direction} movement")
    
    def stop_all_movement(self):
        """Stop all active movements"""
        with self.movement_lock:
            directions_to_stop = list(self.active_movements)
            for direction in directions_to_stop:
                stop_cmd = self.AXES[direction][1]
                self.send_command(stop_cmd)
            self.active_movements.clear()
            print("Stopped all movements")
    
    def move_with_duration(self, duration, *directions):
        """Move in directions for specified duration (like sniffedtest.py)"""
        self.start_movement(*directions)
        time.sleep(duration)
        self.stop_movement(*directions)
    
    def pan_tilt(self, pan_dir, tilt_dir, duration=0.5):
        """Pan and tilt with duration control"""
        movements = []
        
        if pan_dir == -1:  # Left
            movements.append("left")
        elif pan_dir == 1:  # Right
            movements.append("right")
            
        if tilt_dir == -1:  # Up
            movements.append("up")
        elif tilt_dir == 1:  # Down
            movements.append("down")
        
        if movements:
            self.move_with_duration(duration, *movements)
            return True
        return False
    
    def zoom(self, zoom_dir, duration=0.5):
        """Zoom in/out with duration control"""
        if zoom_dir == 1:  # Zoom in
            self.move_with_duration(duration, "zoomin")
            return True
        elif zoom_dir == -1:  # Zoom out
            self.move_with_duration(duration, "zoomout")
            return True
        return False
    
    def focus(self, focus_dir, duration=0.5):
        """Focus in/out with duration control"""
        if focus_dir == 1:  # Focus in (close)
            self.move_with_duration(duration, "focusin")
            return True
        elif focus_dir == -1:  # Focus out (far)
            self.move_with_duration(duration, "focusout")
            return True
        return False
    
    def go_home(self):
        """Go to home position"""
        return self.send_setting("home_go")
    
    def set_home(self):
        """Set current position as home"""
        return self.send_setting("home_set")
    
    def send_setting(self, setting_name):
        """Send a camera setting command"""
        if setting_name in self.SETTINGS:
            success = True
            for cmd in self.SETTINGS[setting_name]:
                # Format command with serial number if needed
                formatted_cmd = cmd.format(serial=self.serial_number) if '{serial}' in cmd else cmd
                if not self.send_command(formatted_cmd):
                    success = False
            return success
        return False
    
    def set_numeric_setting(self, setting_name, value):
        """Set a numeric camera setting"""
        if setting_name in self.NUMERIC_SETTINGS:
            selector = self.NUMERIC_SETTINGS[setting_name]
            path = f"setting?action=setcmd&cmdtype=UVC&selector={selector}&value={value}&UVCID={self.serial_number}"
            return self.send_command(path)
        return False
    
    def set_white_balance(self, mode, value=None):
        """Set white balance mode and optionally value"""
        if mode == "auto":
            return self.send_setting("wb_auto")
        elif mode == "manual":
            success = self.send_setting("wb_manual")
            if success and value is not None:
                success = self.set_numeric_setting("wb", value)
            return success
        return False
    
    def set_low_light(self, enabled):
        """Set low light compensation"""
        return self.send_setting("lowlight_on" if enabled else "lowlight_off")
    
    def set_noise_reduction(self, level):
        """Set noise reduction level (off, low, middle, high)"""
        setting_name = f"noise_reduction_{level}"
        return self.send_setting(setting_name)
    
    def set_mirror(self, enabled):
        """Set mirror/flip mode"""
        return self.send_setting("mirror_true" if enabled else "mirror_false")
    
    def set_saturation(self, value):
        """Set saturation (0-255 typically)"""
        return self.set_numeric_setting("saturation", value)
    
    def set_brightness(self, value):
        """Set brightness (1-9 range)"""
        value = max(1, min(9, int(value)))
        path = f"setting?action=setcmd&cmdtype=UVC&selector=1&value={value}&UVCID={self.serial_number}"
        return self.send_command(path)
    
    def set_sharpness(self, level_value):
        """Set sharpness level (0=off, 1=low, 2=middle, 3=high)"""
        level_value = max(0, min(3, int(level_value)))
        return self.set_numeric_setting("sharpness", level_value)
    
    def set_focus_mode(self, mode):
        """Set focus mode using updated commands"""
        if mode == "auto":
            # Use new direct UVC command: selector=11, value=1 for auto
            path = f"setting?action=setcmd&cmdtype=UVC&selector=11&value=1&UVCID={self.serial_number}"
            return self.send_command(path)
        elif mode == "manual":
            # Use new direct UVC command: selector=11, value=0 for manual
            path = f"setting?action=setcmd&cmdtype=UVC&selector=11&value=0&UVCID={self.serial_number}"
            return self.send_command(path)
        return False
    
    def set_focus_value(self, value):
        """Set manual focus distance (0-255) using updated command"""
        value = max(0, min(255, int(value)))
        # Use new direct UVC command: selector=10, value=0-255
        path = f"setting?action=setcmd&cmdtype=UVC&selector=10&value={value}&UVCID={self.serial_number}"
        return self.send_command(path)
    
    def set_direct_pan_degrees(self, degrees):
        """Set horizontal position in degrees (-169 to 169, 0=center) - API requires integers"""
        degrees = int(max(-169, min(169, degrees)))  # Ensure integer
        path = f"setting?action=setcmd&cmdtype=UVC&selector=7&value={degrees}&UVCID={self.serial_number}"
        success = self.send_command(path)
        if success:
            self.current_pan_deg = degrees
        return success
    
    def set_direct_tilt_degrees(self, degrees):
        """Set vertical position in degrees (-29 to 89, 0=center) - API requires integers"""
        degrees = int(max(-29, min(89, degrees)))  # Ensure integer
        path = f"setting?action=setcmd&cmdtype=UVC&selector=8&value={degrees}&UVCID={self.serial_number}"
        success = self.send_command(path)
        if success:
            self.current_tilt_deg = degrees
        return success
    
    def set_direct_zoom_value(self, zoom_value):
        """Set zoom directly (0-996)"""
        zoom_value = max(0, min(996, int(zoom_value)))
        path = f"setting?action=setcmd&cmdtype=UVC&selector=9&value={zoom_value}&UVCID={self.serial_number}"
        success = self.send_command(path)
        if success:
            self.current_zoom_val = zoom_value
        return success
    
    def get_current_pan_degrees(self):
        """Get current horizontal position - placeholder for now"""
        # TODO: Implement if camera supports getting current position
        return getattr(self, '_current_pan', 0)
    
    def get_current_tilt_degrees(self):
        """Get current vertical position - placeholder for now"""
        # TODO: Implement if camera supports getting current position
        return getattr(self, '_current_tilt', 0)
    
    def get_current_zoom_value(self):
        """Get current zoom value - placeholder for now"""
        # TODO: Implement if camera supports getting current zoom
        return getattr(self, '_current_zoom', 100)