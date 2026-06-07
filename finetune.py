"""
Finetune YOLO to detect only cat and cat_toy.

Prerequisites:
    1. Annotate images in Label Studio with classes: cat, cat_toy
    2. Export as YOLO format zip from Label Studio
    3. Run: python ls_to_yolo.py <export.zip> --out dataset
    4. Run: python finetune.py

The script fine-tunes from a COCO-pretrained checkpoint, so the model
already knows what cats look like — you need far fewer labeled images.
"""

from pathlib import Path

from ultralytics import YOLO

DATASET = "dataset-2/data.yaml"
BASE_MODEL = "yolo11s.pt"   # pretrained on COCO; swap for yolov8s.pt if preferred
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16
PROJECT = "runs/finetune"
NAME = "cattoy"


def main():
    if not Path(DATASET).exists():
        print(
            f"Dataset not found at '{DATASET}'.\n"
            "Run: python ls_to_yolo.py <export.zip> --out dataset"
        )
        return

    model = YOLO(BASE_MODEL)

    model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
    )

    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")
    print(f"To run detection: update detect_objects.py model path to '{best}'")


if __name__ == "__main__":
    main()
