import time

import cv2
from flask import Flask, Response

app = Flask(__name__)

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
time.sleep(2)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_w}x{actual_h}")


def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/stream")
def stream():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return '<img src="/stream" style="max-width:100%">'


if __name__ == "__main__":
    print("Open http://localhost:5000 in your Windows browser")
    app.run(host="0.0.0.0", port=5000)
