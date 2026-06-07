#!/usr/bin/env python3
"""
Auto-label images using YOLO (detection + classification) + SAM2 (segmentation).
YOLO finds where the cat and toy are; SAM2 produces much cleaner polygon masks.

Usage
-----
    python auto_label_sam.py
    python auto_label_sam.py --images images/ --out prelabels_sam.json --conf 0.25
    python auto_label_sam.py --detector yolo11x.pt --sam sam2.1_b.pt   # faster/smaller
"""

import argparse
import json
import uuid
from pathlib import Path

from PIL import Image
from ultralytics import SAM, YOLO

CLASSES = {15: "cat", 32: "toy"}


def to_ls_points(mask_xy, img_w: int, img_h: int) -> list[list[float]]:
    return [
        [round(float(x) / img_w * 100, 3), round(float(y) / img_h * 100, 3)]
        for x, y in mask_xy
    ]


def process_image(detector, sam, img_path: Path, conf: float, doc_root: Path) -> dict:
    img = Image.open(img_path)
    img_w, img_h = img.size

    det = detector(str(img_path), classes=list(CLASSES.keys()), conf=conf, verbose=False)[0]

    result_items = []

    if det.boxes is not None and len(det.boxes):
        boxes_xyxy = det.boxes.xyxy.cpu().tolist()
        cls_ids = det.boxes.cls.cpu().tolist()
        scores = det.boxes.conf.cpu().tolist()

        # SAM segments all boxes in one forward pass
        sam_result = sam(str(img_path), bboxes=boxes_xyxy, verbose=False)[0]

        if sam_result.masks is not None:
            for mask_xy, cls_id, score in zip(sam_result.masks.xy, cls_ids, scores):
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

    rel = img_path.relative_to(doc_root)
    return {
        "data": {"image": f"/data/local-files/?d={rel}"},
        "predictions": [{"model_version": f"{args.detector}+{args.sam}", "result": result_items}],
    }


def main():
    global args
    ap = argparse.ArgumentParser(description="Auto-label with YOLO + SAM2 for Label Studio")
    ap.add_argument("--images", default="images", help="Images directory (default: images/)")
    ap.add_argument("--out", default="prelabels_sam.json", help="Output JSON (default: prelabels_sam.json)")
    ap.add_argument("--detector", default="yolo11x.pt", help="YOLO detection model (default: yolo11x.pt)")
    ap.add_argument("--sam", default="sam2.1_l.pt", help="SAM2 model (default: sam2.1_l.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (default: 0.25)")
    ap.add_argument("--doc-root", default=".", help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (default: .)")
    args = ap.parse_args()

    images_dir = Path(args.images).resolve()
    doc_root = Path(args.doc_root).resolve()
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    print(f"Loading detector : {args.detector}")
    detector = YOLO(args.detector)
    print(f"Loading SAM      : {args.sam}")
    sam = SAM(args.sam)

    tasks = []
    cat_count = toy_count = 0
    for i, img_path in enumerate(image_paths, 1):
        task = process_image(detector, sam, img_path, args.conf, doc_root)
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
    print(f"\nImport {output} into Label Studio the same way as prelabels.json.")


if __name__ == "__main__":
    main()
