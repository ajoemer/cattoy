from ultralytics import YOLO

# After finetuning, replace with: runs/finetune/cattoy/weights/best.pt
MODEL_PATH = "yolo26s.pt"

# When using the base COCO model, filter to cat only (class 15).
# Set to None after finetuning — the custom model will only have your classes.
FILTER_CLASSES = []

model = YOLO(MODEL_PATH)

filter_ids = (
    [k for k, v in model.names.items() if v in FILTER_CLASSES]
    if FILTER_CLASSES
    else None
)

results = model("/root/personal/cattoy/images/20260420_032126_916399.jpg", classes=filter_ids)

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
