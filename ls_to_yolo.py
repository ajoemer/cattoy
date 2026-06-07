"""
Convert a Label Studio YOLO export zip into Ultralytics training structure.

Usage:
    python ls_to_yolo.py <export.zip> [--out dataset]

Label Studio export steps:
    1. Open your project → Export → YOLO format → download the zip
    2. Run this script on that zip

Output layout:
    dataset/
      images/train/   images/val/
      labels/train/   labels/val/
      data.yaml
"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path


def convert(
    zip_path: str, out_dir: str = "dataset", val_split: float = 0.2, seed: int = 42,
    images_dir: str = None,
):
    zip_path = Path(zip_path)
    out = Path(out_dir)

    # Extract zip
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)

    # Locate images and labels inside the extracted folder
    images = sorted(
        (tmp / "images").glob("*") if (tmp / "images").exists() else tmp.rglob("*.jpg")
    )
    labels_dir = tmp / "labels" if (tmp / "labels").exists() else None

    # Fall back to an external images directory (e.g. when Label Studio local-file
    # serving is used and the export ZIP has an empty images/ folder)
    if not images and images_dir:
        ext_dir = Path(images_dir)
        images = sorted(ext_dir.glob("*.jpg")) + sorted(ext_dir.glob("*.png"))

    if not images:
        print("No images found in export. Use --images-dir to point at your images folder.")
        return

    # Train/val split
    random.seed(seed)
    random.shuffle(images)
    split = int(len(images) * (1 - val_split))
    splits = {"train": images[:split], "val": images[split:]}

    for split_name, imgs in splits.items():
        (out / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, out / "images" / split_name / img.name)
            if labels_dir:
                lbl = labels_dir / (img.stem + ".txt")
                if lbl.exists():
                    shutil.copy(lbl, out / "labels" / split_name / lbl.name)

    # Read classes from classes.txt if present
    classes_file = next(tmp.rglob("classes.txt"), None)
    class_names = []
    if classes_file:
        class_names = [
            l.strip() for l in classes_file.read_text().splitlines() if l.strip()
        ]

    # Write data.yaml
    yaml_content = f"""path: {out.resolve()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
    (out / "data.yaml").write_text(yaml_content)
    shutil.rmtree(tmp)

    print(f"Done. Dataset written to: {out}/")
    print(f"  train: {len(splits['train'])} images")
    print(f"  val:   {len(splits['val'])} images")
    print(f"  classes ({len(class_names)}): {class_names}")
    print(f"\nTo train:\n  yolo train model=yolov8s.pt data={out}/data.yaml epochs=50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label Studio YOLO export → Ultralytics dataset"
    )
    parser.add_argument("zip", help="Path to Label Studio YOLO export zip")
    parser.add_argument(
        "--out", default="dataset", help="Output directory (default: dataset)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation fraction (default: 0.2)",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Fallback image directory when the export ZIP has an empty images/ folder "
             "(common with Label Studio local-file serving). E.g. --images-dir images/",
    )
    args = parser.parse_args()
    convert(args.zip, args.out, args.val_split, images_dir=args.images_dir)
