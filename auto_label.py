#!/usr/bin/env python3
"""
Auto-label images with a YOLO segmentation model and export Label Studio
pre-annotations JSON.  Cat (COCO class 15) and sports ball (COCO class 32,
used as a proxy for the cat toy) are detected as polygon masks.

Usage
-----
    python auto_label.py
    python auto_label.py --images images/ --out prelabels.json --conf 0.25
    python auto_label.py --model yolov8x-seg.pt   # fall back to v8 if v11 fails
"""

import argparse
import json
import uuid
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

# COCO class id → Label Studio label name
CLASSES = {15: "cat", 32: "toy"}

LS_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2" pointSize="small">
    <Label value="cat" background="#FF0000"/>
    <Label value="toy" background="#00AAFF"/>
  </PolygonLabels>
</View>"""


def to_ls_points(mask_xy, img_w: int, img_h: int) -> list[list[float]]:
    """Convert absolute mask polygon to Label Studio % coordinates (0-100)."""
    return [
        [round(float(x) / img_w * 100, 3), round(float(y) / img_h * 100, 3)]
        for x, y in mask_xy
    ]


def process_image(model, img_path: Path, conf: float, doc_root: Path) -> dict:
    img = Image.open(img_path)
    img_w, img_h = img.size

    results = model(
        str(img_path),
        classes=list(CLASSES.keys()),
        conf=conf,
        verbose=False,
    )

    result_items = []
    for r in results:
        if r.masks is None:
            continue
        for mask_xy, cls_id, score in zip(r.masks.xy, r.boxes.cls, r.boxes.conf):
            points = to_ls_points(mask_xy, img_w, img_h)
            if len(points) < 3:
                continue
            result_items.append({
                "id": uuid.uuid4().hex[:8],
                "type": "polygonlabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": img_w,
                "original_height": img_h,
                "image_rotation": 0,
                "value": {
                    "points": points,
                    "polygonlabels": [CLASSES[int(cls_id)]],
                    "closed": True,
                },
                "score": round(float(score), 3),
            })

    # Path relative to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
    rel = img_path.relative_to(doc_root)
    image_url = f"/data/local-files/?d={rel}"

    return {
        "data": {"image": image_url},
        "predictions": [{"model_version": args.model, "result": result_items}],
    }


def main():
    global args
    ap = argparse.ArgumentParser(description="Auto-label images for Label Studio import")
    ap.add_argument("--images", default="images", help="Images directory (default: images/)")
    ap.add_argument("--out", default="prelabels.json", help="Output JSON file (default: prelabels.json)")
    ap.add_argument("--model", default="yolo11x-seg.pt", help="YOLO seg model (default: yolo11x-seg.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25)")
    ap.add_argument("--doc-root", default=".", help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (default: .)")
    args = ap.parse_args()

    images_dir = Path(args.images)
    doc_root = Path(args.doc_root).resolve()
    images_dir_abs = images_dir.resolve()

    image_paths = sorted(images_dir_abs.glob("*.jpg")) + sorted(images_dir_abs.glob("*.png"))
    if not image_paths:
        print(f"No images found in {images_dir_abs}")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    tasks = []
    cat_count = toy_count = 0
    for i, img_path in enumerate(image_paths, 1):
        task = process_image(model, img_path, args.conf, doc_root)
        results = task["predictions"][0]["result"]
        n_cat = sum(1 for r in results if r["value"]["polygonlabels"] == ["cat"])
        n_toy = sum(1 for r in results if r["value"]["polygonlabels"] == ["toy"])
        cat_count += n_cat
        toy_count += n_toy
        print(f"[{i:3}/{len(image_paths)}] {img_path.name}  cat={n_cat}  toy={n_toy}")
        tasks.append(task)

    output = Path(args.out)
    output.write_text(json.dumps(tasks, indent=2))

    print(f"\n{'─' * 50}")
    print(f"  Images processed : {len(tasks)}")
    print(f"  cat annotations  : {cat_count}")
    print(f"  toy annotations  : {toy_count}")
    print(f"  Output           : {output}")
    print(f"{'─' * 50}")
    print()
    print("Label Studio project labeling config (copy into Labeling Setup > Code):")
    print()
    print(LS_CONFIG)
    print()
    print("How to import into Label Studio:")
    print()
    print("  1. Start Label Studio with local file serving enabled:")
    print(f"       LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\")
    print(f"       LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={doc_root} \\")
    print(f"       label-studio start")
    print()
    print("  2. Create a project → Labeling Setup → paste the XML config above")
    print()
    print("  3. Settings → Cloud Storage → Add Source Storage")
    print("     Type: Local files")
    print(f"     Absolute local path: {images_dir_abs}")
    print("     Toggle 'Treat every bucket object as a source file' ON")
    print("     Click 'Sync Storage'")
    print()
    print(f"  4. Project → Import → upload {output}")
    print()
    print("  Note: toy (ball) detection uses COCO 'sports ball' class.")
    print("  Small or unusual cat toys may be missed — correct those in Label Studio.")


if __name__ == "__main__":
    main()
