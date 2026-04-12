import os
import time

import cv2

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

# Set a common format explicitly
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)  # give the camera time to initialize

ret, frame = None, None
for i in range(10):
    ret, frame = cap.read()
    if ret:
        break
    print(f"Attempt {i+1} failed, retrying...")
    time.sleep(0.5)

if ret:
    cv2.imwrite("test.jpg", frame)
    print("Success! Frame saved as test.jpg")
else:
    print("Failed to capture frame after 10 attempts")

cap.release()
