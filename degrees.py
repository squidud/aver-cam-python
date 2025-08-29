import requests
import time
import threading

urlh = "http://localhost:36680/setting?action=setcmd&cmdtype=UVC&selector=7&value=00&UVCID=5203561500051"
urlv = "http://localhost:36680/setting?action=setcmd&cmdtype=UVC&selector=9&value=0&UVCID=5203561500051"
try:
    response = requests.get(urlh, timeout=2)
    print(f"Sent {urlh}: {response.status_code}")
    response = requests.get(urlv, timeout=2)
    print(f"Sent {urlv}: {response.status_code}")

except requests.RequestException as e:
    print(f"Error sending {urlh}: {e}")
    print(f"Error sending {urlv}: {e}")

# URL Format for degree control:
# http://localhost:36680/setting?action=setcmd&cmdtype=UVC&selector=8&value=<int value here in degrees>&UVCID=5203561500051 (Vertical) (keep original command for ptz control buttons, but use this for things like auto tracking)
# http://localhost:36680/setting?action=setcmd&cmdtype=UVC&selector=7&value=<int value here in degrees>&UVCID=5203561500051 (Horizontal) (keep original command for ptz control buttons, but use this for things like auto tracking)
# for degrees, 0 is center ('home'), positive is right/down, negative is left/up
# horizontal range is -169 through 169 (in degrees)
# vertical range is -29 throu 89 (in degrees)

# Direct zoom control:
# http://localhost:36680/setting?action=setcmd&cmdtype=UVC&selector=9&value=<int value 0-996>&UVCID=5203561500051 (keep original command for ptz control buttons, but use this for things like auto tracking)

#focus control: 
# http://localhost:36680/setting?action=setcmd&cmdtype=UVC&selector=11&value=<1 for auto, 0 for manual>&UVCID=5203561500051 (auto focus enable/disable) (rebind gui to these commands)
# http://localhost:36680/setting?action=setcmd&cmdtype=UVC&selector=10&value=<int value 0-255>&UVCID=5203561500051" (manual focus distance) (rebind gui to these commands)

# saturation range is ONLY 1-9 (update gui accordingly)
# sharpness range is ONLY off, low, middle, and high (update gui accordingly))

# ANY of these commands can be used back to back with no delay, and they will be done simultaneously (or as close to it as the camera can go).

# Also add small warning to bottom of the gui that says "PTZApp 2 MUST be running for PTZ controls to work"