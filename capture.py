"""
Capture images from the webcam for labelling.

Opens a live preview in the browser. Click the Capture button (or press
Space / Enter on the page) to save a timestamped JPEG to the output folder.

Usage:
    python capture.py [--out images] [--port 5001]
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
time.sleep(2)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_w}x{actual_h}")

output_dir = Path("images")
latest_frame = None


def generate_frames():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        latest_frame = frame.copy()
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


@app.route("/capture", methods=["POST"])
def capture():
    if latest_frame is None:
        return jsonify({"error": "No frame available"}), 500

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    path = output_dir / filename
    cv2.imwrite(str(path), latest_frame)
    count = len(list(output_dir.glob("*.jpg")))
    print(f"Saved: {path}")
    return jsonify({"saved": str(path), "total": count})


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html>
<head>
  <title>Image Capture</title>
  <style>
    body { margin: 0; background: #111; display: flex; flex-direction: column;
           align-items: center; justify-content: center; min-height: 100vh;
           font-family: sans-serif; color: #fff; }
    img  { max-width: 100%; border: 2px solid #444; }
    button { margin-top: 16px; padding: 12px 40px; font-size: 1.2rem;
             background: #e74c3c; color: #fff; border: none; border-radius: 6px;
             cursor: pointer; }
    button:active { background: #c0392b; }
    #status { margin-top: 10px; font-size: 0.95rem; color: #aaa; min-height: 1.4em; }
  </style>
</head>
<body>
  <img src="/stream">
  <button id="btn" onclick="capture()">Capture (Space)</button>
  <div id="status"></div>
  <script>
    async function capture() {
      const res  = await fetch("/capture", { method: "POST" });
      const data = await res.json();
      if (data.saved) {
        document.getElementById("status").textContent =
          "Saved: " + data.saved + "  |  Total: " + data.total;
      } else {
        document.getElementById("status").textContent = "Error: " + data.error;
      }
    }
    document.addEventListener("keydown", e => {
      if (e.code === "Space" || e.code === "Enter") { e.preventDefault(); capture(); }
    });
  </script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam capture tool for labelling")
    parser.add_argument(
        "--out", default="images", help="Output folder (default: images)"
    )
    parser.add_argument("--port", type=int, default=5001, help="Port (default: 5001)")
    args = parser.parse_args()

    output_dir = Path(args.out)
    print(f"Saving images to: {output_dir.resolve()}")
    print(f"Open http://localhost:{args.port} in your Windows browser")
    app.run(host="0.0.0.0", port=args.port)
