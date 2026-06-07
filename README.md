# cattoy — Object Detection with YOLO

## Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Capturing Images

Use `capture.py` to take photos from your webcam directly into an `images/` folder ready for labelling.

```bash
python capture.py
```

Then open `http://localhost:5001` in your browser. A live preview of the camera will appear.

- Click the **Capture** button, or press **Space / Enter**, to save a photo
- Each image is saved as a timestamped JPEG (e.g. `20260416_143201_123456.jpg`)
- The status bar shows the saved path and running total

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `images` | Folder to save images into |
| `--port` | `5001` | Port to serve the preview on |

Example saving to a custom folder:

```bash
python capture.py --out raw_images --port 5002
```

Once done, upload the `images/` folder into Label Studio (see below).

---

## Labelling Images with Label Studio

### 0. Auto-generate Pre-annotations (Recommended)

Instead of labelling 330 images by hand, run `auto_label.py` first to generate
polygon masks with a YOLO segmentation model. You then only need to **correct**
the results in Label Studio.

```bash
python auto_label.py
```

This detects **cat** (COCO class 15, very reliable) and **toy / sports ball**
(COCO class 32, works for round balls — unusual shapes may need manual labels).
Output is `prelabels.json`.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--images` | `images` | Folder with your captured photos |
| `--out` | `prelabels.json` | Output JSON for Label Studio import |
| `--model` | `yolo11x-seg.pt` | YOLO segmentation model (auto-downloaded) |
| `--conf` | `0.25` | Detection confidence threshold |
| `--doc-root` | `.` | Must match `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` |

### 1. Start Label Studio with Local File Serving

```bash
source .venv/bin/activate
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd) \
label-studio start
```

Opens at `http://localhost:8080`. Create an account on first run.

### 2. Create a Project with Polygon Segmentation

1. Click **Create Project** and give it a name
2. Go to the **Labeling Setup** tab → click **Code** in the top-right
3. Paste this XML config and click **Save**:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2" pointSize="small">
    <Label value="cat" background="#FF0000"/>
    <Label value="toy" background="#00AAFF"/>
  </PolygonLabels>
</View>
```

### 3. Connect Your Images via Local Storage

1. Go to your project → **Settings** → **Cloud Storage**
2. Click **Add Source Storage** → choose **Local files**
3. Set **Absolute local path** to your `images/` folder (e.g. `/root/personal/cattoy/images`)
4. Toggle **"Treat every bucket object as a source file"** ON
5. Click **Add Storage** then **Sync Storage**

### 4. Import Pre-annotations

1. Go to your project → **Import**
2. Upload `prelabels.json`
3. Click **Import**

Label Studio will match the pre-annotations to the synced images.

### 5. Review and Correct

1. Open an image from the project list
2. Inspect the auto-generated polygon masks
3. Add missing objects, delete false positives, or adjust polygon points
4. Click **Submit** to move to the next image

### 6. Export Annotations

1. Go to your project → **Export**
2. Select **YOLO** format
3. Click **Export** and save the downloaded zip file

---

## Converting to Ultralytics Format

Run `ls_to_yolo.py` on the exported zip to produce a training-ready dataset:

```bash
python ls_to_yolo.py export.zip --out dataset
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `dataset` | Output directory |
| `--val-split` | `0.2` | Fraction of images used for validation |

Output structure:

```
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

---

## Training

```bash
yolo train model=yolov8s.pt data=dataset/data.yaml epochs=50
```

---

## Running Detection

**On a single image:**

```bash
python detect_objects.py
```

**Live stream (requires webcam at `/dev/video0`):**

```bash
python detect_stream.py
```

Then open `http://localhost:5000` in your browser.
