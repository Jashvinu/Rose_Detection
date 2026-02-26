# Rose Disease Detection — Inference Server

Minimal FastAPI server. Send an image, get back detections. The monolith handles everything else.

---

## API

### `POST /detect`

**Request** — multipart form

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | file | required | Any common image format |
| `confidence_threshold` | float | `0.5` | Override per-request |

**Response**

```json
{
  "detections": [
    {
      "label": "powdery_mildew_leaf",
      "confidence": 0.87,
      "bbox": { "x1": 120.4, "y1": 45.1, "x2": 310.8, "y2": 280.3 }
    }
  ]
}
```

Bounding box coordinates are in original image pixel space.

An empty `detections` list means the model found nothing above the confidence threshold.

### `GET /health`

```json
{
  "status": "ok",
  "model": "YOLO11s TFLite",
  "labels": ["healthy_leaf", "powdery_mildew_leaf", "two_spotted_spider_mite_damage_leaf", "unknown_disease_leaf", "chemical_residue_leaf", "downy_mildew_leaf"]
}
```

---

## Classes

| Label | Description |
|---|---|
| `healthy_leaf` | No disease detected |
| `powdery_mildew_leaf` | Powdery mildew infection |
| `two_spotted_spider_mite_damage_leaf` | Spider mite damage |
| `unknown_disease_leaf` | Unidentified disease |
| `chemical_residue_leaf` | Chemical residue present |
| `downy_mildew_leaf` | Downy mildew infection |

---

## Run

```bash
docker compose build
docker compose up
```

Server runs at `http://localhost:8000`.

---

## Model

| Property | Value |
|---|---|
| Architecture | YOLO11s |
| Format | TFLite float32 |
| Input | `(1, 640, 640, 3)` normalised [0, 1] |
| Output | `(1, 10, 8400)` — 4 box coords + 6 class scores |

---

## Configuration

Override via environment variables:

```bash
ROSE_CONFIDENCE_THRESHOLD=0.4
ROSE_NMS_IOU_THRESHOLD=0.45
```
