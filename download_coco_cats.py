"""
Download only cat images from COCO and export them in YOLO format.

The output merges cleanly with toy images labeled via Label Studio,
so you end up with a single dataset: cat (class 0) + cat_toy (class 1).

Install deps first:
    pip install fiftyone

Usage:
    python download_coco_cats.py [--max-samples 2000] [--out dataset]
"""

import argparse
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
import yaml


def download_and_export(max_samples: int, out_dir: str):
    out = Path(out_dir)

    print(f"Downloading up to {max_samples} cat images from COCO...")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=["cat"],
        max_samples=max_samples,
    )

    # Keep only cat detections (images may have other objects annotated too)
    view = dataset.filter_labels("ground_truth", fo.ViewField("label") == "cat")

    tmp_export = out / "_coco_export"
    tmp_export.mkdir(parents=True, exist_ok=True)

    print("Exporting to YOLO format...")
    view.export(
        export_dir=str(tmp_export),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=["cat"],
    )

    # Merge into existing dataset structure (images/train, labels/train)
    # Cat becomes class 0; your toy labels (class 1) come from Label Studio
    for split in ("train", "val"):
        src_images = tmp_export / "images" / split
        src_labels = tmp_export / "labels" / split
        if not src_images.exists():
            continue

        dst_images = out / "images" / split
        dst_labels = out / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for img in src_images.iterdir():
            shutil.copy(img, dst_images / img.name)

        for lbl in src_labels.iterdir():
            # Remap class index to 0 (cat) in case fiftyone writes something else
            lines = lbl.read_text().splitlines()
            remapped = [f"0 {' '.join(l.split()[1:])}" for l in lines if l.strip()]
            (dst_labels / lbl.name).write_text("\n".join(remapped) + "\n")

    shutil.rmtree(tmp_export)

    # Write data.yaml only if it doesn't exist yet (toy class added later)
    yaml_path = out / "data.yaml"
    if not yaml_path.exists():
        data = {
            "path": str(out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 2,
            "names": ["cat", "cat_toy"],
        }
        yaml_path.write_text(yaml.dump(data, default_flow_style=False))
        print(f"Created {yaml_path}")
    else:
        print(f"{yaml_path} already exists — skipped (keeping your existing config)")

    print(f"\nDone. COCO cat images saved to: {out}/")
    print("Next steps:")
    print("  1. Label your cat_toy images in Label Studio (class name: cat_toy)")
    print("  2. Export as YOLO zip and run: python ls_to_yolo.py <export.zip> --out dataset")
    print("     (ls_to_yolo will merge into the same dataset/ folder)")
    print("  3. Train: python finetune.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-samples", type=int, default=2000, help="Max cat images to download"
    )
    parser.add_argument("--out", default="dataset", help="Output dataset directory")
    args = parser.parse_args()
    download_and_export(args.max_samples, args.out)
