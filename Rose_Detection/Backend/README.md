# Rose Disease Detection — Inference Server

Minimal FastAPI inference server for rose leaf disease detection. Send an image, get back detections. The monolith handles record updates, condition logic, and everything else.

Interactive API docs available at `http://localhost:8000/docs` once running.

---

## Quick Start

```bash
docker compose build
docker compose up
```

Server runs at `http://localhost:8000`.

---

## API

### `POST /detect`

Send one image, receive a list of detections.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | file | yes | — | JPEG, PNG, or any PIL-supported format |
| `confidence_threshold` | float | no | `0.5` | Override detection confidence for this request |

**Example — curl**

```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@/path/to/leaf.jpg"
```

With a custom confidence threshold:

```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@/path/to/leaf.jpg" \
  -F "confidence_threshold=0.4"
```

**Response**

```json
{
  "detections": [
    {
      "label": "powdery_mildew_leaf",
      "confidence": 0.87,
      "bbox": { "x1": 120.4, "y1": 45.1, "x2": 310.8, "y2": 280.3 }
    },
    {
      "label": "healthy_leaf",
      "confidence": 0.91,
      "bbox": { "x1": 400.0, "y1": 60.2, "x2": 590.5, "y2": 310.7 }
    }
  ]
}
```

- Bounding box coordinates are in **original image pixel space** (not normalised)
- Multiple detections per image are possible
- Empty `detections` list = nothing found above the confidence threshold

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "YOLO11s TFLite",
  "labels": [
    "healthy_leaf",
    "powdery_mildew_leaf",
    "two_spotted_spider_mite_damage_leaf",
    "unknown_disease_leaf",
    "chemical_residue_leaf",
    "downy_mildew_leaf"
  ]
}
```

---

## Classes

| Label | Training Images |
|---|---|
| `healthy_leaf` | 10,676 |
| `powdery_mildew_leaf` | 2,705 |
| `two_spotted_spider_mite_damage_leaf` | 3,056 |
| `unknown_disease_leaf` | 3,248 |
| `chemical_residue_leaf` | 2,533 |
| `downy_mildew_leaf` | — |

---

## Model

| Property | Value |
|---|---|
| Architecture | YOLO11s |
| Format | TFLite float32 |
| Input | `(1, 640, 640, 3)` — normalised [0, 1] |
| Output | `(1, 10, 8400)` — 4 box coords + 6 class scores |
| Post-processing | Confidence filter → greedy NMS per class |

---

## Configuration

All settings can be overridden with `ROSE_` environment variables:

| Variable | Default | Description |
|---|---|---|
| `ROSE_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence to include a detection |
| `ROSE_NMS_IOU_THRESHOLD` | `0.45` | IoU threshold for non-max suppression |

Example — set in `docker-compose.yml`:

```yaml
services:
  rose-detector:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ROSE_CONFIDENCE_THRESHOLD=0.4
```

---

## Local Development (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Requires Python 3.11. `tflite-runtime` is Linux x86-64 only — use Docker on macOS/Windows.

---

## Project Structure

```
Backend/
├── app/
│   ├── main.py               # FastAPI app — /detect and /health
│   ├── config.py             # Settings via env vars (ROSE_ prefix)
│   ├── schemas.py            # BBox, Detection, DetectResponse
│   ├── model/
│   │   ├── rose_disease_model.tflite
│   │   └── labels.txt
│   └── pipeline/
│       ├── classifier.py     # YOLOTFLiteDetector — runs TFLite inference
│       └── tiler.py          # compute_iou, nms_per_class
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── requirements.txt
```
