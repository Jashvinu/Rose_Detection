# Rose Disease Detection — Backend

FastAPI inference server for rose leaf disease detection using a YOLO11s model exported to TFLite. Runs a full **detect → track → count** pipeline across sequential images.

---

## Model

| Property | Value |
|---|---|
| Architecture | YOLO11s |
| Format | TFLite float32 |
| Input | `(1, 640, 640, 3)` float32, normalised [0, 1] |
| Output | `(1, 10, 8400)` — 4 box coords + 6 class scores |
| File | `app/model/rose_disease_model.tflite` |

### Classes

| ID | Label |
|---|---|
| 0 | `healthy_leaf` |
| 1 | `powdery_mildew_leaf` |
| 2 | `two_spotted_spider_mite_damage_leaf` |
| 3 | `unknown_disease_leaf` |
| 4 | `chemical_residue_leaf` |
| 5 | `downy_mildew_leaf` |

---

## Pipeline

```
POST /api/v1/detect (images)
  │
  ├─ per frame:
  │    └─ YOLOTFLiteDetector.detect()
  │         ├─ resize to 640×640, normalise /255
  │         ├─ TFLite inference → (1, 10, 8400)
  │         ├─ argmax over 6 class channels → label + confidence
  │         ├─ scale boxes back to original pixel coords
  │         └─ confidence filter
  │    └─ NMS per class (greedy IoU)
  │    └─ MotionCompensator (ORB + RANSAC homography)
  │    └─ IoUTracker (cross-frame deduplication)
  │
  └─ finalize:
       └─ majority-vote class label per track
       └─ PlantVillage affliction mapping
       └─ annotated JPEG (base64) per frame
```

---

## Quick Start

### Docker (recommended)

```bash
cd Backend
docker compose build
docker compose up
```

Server runs at `http://localhost:8000`.

### Local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## API

### `POST /api/v1/detect`

Upload one or more images and get disease detections with track-level deduplication.

**Form fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `images` | files | required | Sequential images (multipart) |
| `token` | string | `""` | PV token: `scout:1\|partition:2\|bed:3` |
| `callback_url` | string | `""` | POST results here when done |
| `confidence_threshold` | float | `0.5` | Override detection confidence |
| `min_hits` | int | `3` | Frames a track must appear before counted |
| `visualize` | bool | `true` | Embed annotated JPEG (base64) in response |
| `motion_compensation` | bool | `true` | ORB-based cross-frame stabilisation |

**Response** (`SummaryResponse`)

```json
{
  "job_id": "uuid",
  "status": "completed",
  "summary": {
    "powdery_mildew_leaf": 2,
    "healthy_leaf": 1,
    "total_detections": 3
  },
  "per_frame_details": [
    {
      "frame_index": 0,
      "filename": "img1.jpg",
      "detections": [
        {
          "bbox": {"x1": 120, "y1": 45, "x2": 310, "y2": 280},
          "label": "powdery_mildew_leaf",
          "confidence": 0.87,
          "frame_index": 0
        }
      ],
      "active_tracks": 1,
      "annotated_image": "data:image/jpeg;base64,..."
    }
  ],
  "unique_detections": [...],
  "afflictions": [{"id": 2, "name": "Powdery Mildew"}],
  "condition": "new"
}
```

### `GET /api/v1/jobs/{job_id}`

Retrieve stored results for a completed job.

### `GET /api/v1/jobs/{job_id}/frames/{index}/image`

Returns the annotated JPEG for a specific frame. Open directly in a browser.

### `GET /health`

```json
{
  "status": "ok",
  "model": "YOLO11s TFLite",
  "model_file": "rose_disease_model.tflite",
  "labels": ["healthy_leaf", "powdery_mildew_leaf", ...]
}
```

---

## PlantVillage Affliction Mapping

| Model label | Affliction ID | Name |
|---|---|---|
| `healthy_leaf` | — | (no affliction) |
| `downy_mildew_leaf` | 1 | Downy Mildew |
| `powdery_mildew_leaf` | 2 | Powdery Mildew |
| `two_spotted_spider_mite_damage_leaf` | 3 | Mite |
| `unknown_disease_leaf` | 5 | Unknown Disease |
| `chemical_residue_leaf` | 6 | Chemical Residue |

---

## Configuration

All settings can be overridden with `ROSE_` environment variables:

```bash
ROSE_CONFIDENCE_THRESHOLD=0.4
ROSE_NMS_IOU_THRESHOLD=0.45
ROSE_MIN_HITS=3
ROSE_MOTION_COMPENSATION=true
```

---

## Project Structure

```
Backend/
├── app/
│   ├── main.py                  # FastAPI app + lifespan model load
│   ├── config.py                # Settings (pydantic-settings)
│   ├── schemas.py               # Pydantic data models
│   ├── model/
│   │   ├── rose_disease_model.tflite
│   │   └── labels.txt
│   └── pipeline/
│       ├── classifier.py        # YOLOTFLiteDetector (TFLite inference)
│       ├── tiler.py             # compute_iou, nms_per_class, detect_in_image
│       ├── orchestrator.py      # Wires detector → tracker → counter
│       ├── tracker.py           # IoU-based cross-frame tracker
│       ├── counter.py           # Majority-vote track finalisation
│       ├── motion.py            # ORB + RANSAC motion compensation
│       ├── visualizer.py        # Bounding-box annotation
│       └── affliction_mapper.py # Label → PlantVillage affliction ID
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── requirements.txt
```
