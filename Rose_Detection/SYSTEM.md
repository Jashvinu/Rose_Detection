# Rose Detection Backend — System Documentation

## What We're Building

FastAPI backend that accepts an S3 image URI, runs the TFLite model, and returns detections for a single image.

## What We Built

### Project Structure

```
Rose_Detection/
├── Backend/                     # FastAPI detection server
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── postman_collection.json
│   └── app/
│       ├── main.py              # FastAPI app + endpoints
│       ├── config.py            # Settings (env-overridable via ROSE_*)
│       ├── schemas.py           # Pydantic models
│       ├── model/
│       │   ├── rose_disease_model.tflite
│       │   └── labels.txt
│       └── pipeline/
│           ├── classifier.py    # TFLite model wrapper
│           └── tiler.py         # NMS utilities
├── PLANTVILLAGE_API_REFERENCE.md
└── SYSTEM.md
```

### Detection Pipeline

```
Input: S3 image URI
            │
            ▼
   ┌───────────────────┐
   │   TFLite Model    │  Run inference on single image
   │ (classifier.py)   │
   └─────────┬─────────┘
             │ detections
             ▼
       JSON Response
```

### API Endpoints

#### Detection

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/detect_flowers` | Fetch S3 image, run model, return detections |
| `GET` | `/health` | Server + model status |

#### POST /detect_flowers — JSON Body

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_s3_uri` | string | required | S3 URI for the image to analyze |

#### Response

```json
{
  "detections": [
    {
      "label": "powdery_mildew_leaf",
      "confidence": 0.87,
      "bbox": {"x1": 120.4, "y1": 45.1, "x2": 310.8, "y2": 280.3}
    }
  ]
}
```

---

## Configuration

All settings are overridable via `ROSE_` environment variables:

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `confidence_threshold` | 0.5 | `ROSE_CONFIDENCE_THRESHOLD` | Min confidence to keep a detection |
| `nms_iou_threshold` | 0.45 | `ROSE_NMS_IOU_THRESHOLD` | NMS suppression threshold |

---

## Running

### Docker (recommended)

```bash
cd Backend
docker compose up --build
```

Server starts at `http://localhost:8000`.

### Interactive Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Testing with Postman

Import `Backend/postman_collection.json` and run the `POST /detect_flowers` request.
