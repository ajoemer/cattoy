# Adding New Data to Existing Dataset

Use this when you want to add new images (positive or negative) to an existing Label Studio project without losing previous annotations.

## 1. Capture new images

```bash
python capture.py --out images/
```

Open http://localhost:5001 in your Windows browser and capture frames.
Images must go into `images/` (or wherever Label Studio local storage is configured to serve).

## 2. Generate pre-annotations for the new images only

Put the new images in a temp folder, run SAM on just those, then move them into `images/`:

```bash
# Run SAM only on new images
python auto_label_sam.py --images <folder-with-new-images>/ --out prelabels_new.json --doc-root .

# Move new images into the main images folder
mv <folder-with-new-images>/*.jpg images/
```

For **negative images** (no cat/toy present), SAM will produce empty `result: []` automatically — no manual work needed.

> **Important:** `--doc-root` must match `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` used when starting Label Studio, otherwise image URLs will break.

## 3. Import into existing Label Studio project

1. Open your existing project in Label Studio
2. Click **Import**
3. Upload `prelabels_new.json`

Label Studio adds new tasks without touching existing annotations.

## 4. Annotate / submit the new tasks

- **Negative images:** Open each task and click **Submit** with no annotations drawn — empty submission = background sample.
- **Positive images:** Review the SAM pre-annotations, correct any mistakes, then Submit.

## 5. Export the full project

Project → **Export** → **YOLO format** → download the zip.

## 6. Convert to Ultralytics dataset

```bash
python ls_to_yolo.py <export.zip> --out dataset-1 --val-split 0.2 --images-dir images/
```

This overwrites the previous `dataset-1/` with all data (old + new).

## 7. Retrain

```bash
python finetune.py
```

Best weights will be saved to `runs/finetune/cattoy/weights/best.pt`.
