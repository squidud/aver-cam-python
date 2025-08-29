#!/usr/bin/env python3
"""
Simple viewer for CamControl Virtual Camera
This script reads from the shared memory and displays the video.
"""
import cv2
import numpy as np
import mmap
import struct
import time
import tempfile
import os

def main():
    shared_filename = os.path.join(tempfile.gettempdir(), "camcontrol_virtual_camera.dat")
    
    if not os.path.exists(shared_filename):
        print(f"Virtual camera file not found: {shared_filename}")
        print("Make sure CamControl virtual camera is running.")
        return
    
    print(f"Reading from virtual camera: {shared_filename}")
    print("Press 'q' to quit")
    
    try:
        with open(shared_filename, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                while True:
                    # Read metadata
                    mm.seek(0)
                    metadata = mm.read(16)
                    if len(metadata) < 16:
                        time.sleep(0.033)  # ~30 fps
                        continue
                    
                    width, height, ts_high, ts_low = struct.unpack('IIII', metadata)
                    
                    if width == 0 or height == 0:
                        time.sleep(0.033)
                        continue
                    
                    # Read frame data
                    frame_size = width * height * 3
                    mm.seek(16)
                    frame_data = mm.read(frame_size)
                    
                    if len(frame_data) != frame_size:
                        time.sleep(0.033)
                        continue
                    
                    # Convert to numpy array
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = frame.reshape((height, width, 3))
                    
                    # Display frame
                    cv2.imshow('CamControl Virtual Camera', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
