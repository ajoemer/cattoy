from ultralytics import YOLO

model = YOLO("yolo26s.pt")

results = model("test.jpg")

for result in results:
    result.show()
    result.save(filename="result.jpg")

    boxes = result.boxes
    print(f"Detected {len(boxes)} object(s):")
    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        print(f"  {label}: {confidence:.2f} confidence at {coords}")
