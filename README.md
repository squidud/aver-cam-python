# Controlling Aver (AVer) PTZ Conference Cameras (Tested on CAM520 Pro) With Python and HTTP

### Also includes GUI control application with live face tracking. :) Only USB is needed, no ethernet or RS232.

## How it works:
These cameras are super locked down. I couldn't figure out how to send PTZ commands over any protocols without advanced network configurations or expensive RS232 adapters. 
After using AVer's PTZApp 2, I noticed that it sent HTTP requests over the network, and I reverse-engineered these requests to create web-based commands myself. 

## Important:
### 1: PTZApp 2 MUST be installed and running for any movement-based camera controls to work
### 2: Find the localhost port in the URL of the PTZApp 2 GUI, and find the serial number in the PTZApp information section
### 3: I have only tested the app with my 1 CAM520 Pro, but I would ASSUME it will work with other models
### 4: This app is not perfect. It's a hacky workaround for a stupidly-closed source system. Idk man, I can only do so much
### 5: This app was made for windows (hence the batch files) but everything should technically be cross compatible if you know how to use python commands and stuff.
### 6: Install OBS to have the virtual webcam work, it uses OBS Virtual Camera to function, soz :/
