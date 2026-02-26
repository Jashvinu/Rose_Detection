# PlantVillage Rose Edition — System Documentation

## What We're Building

An automated rose disease detection system for PlantVillage field scouts. The backend accepts an S3 image URI, runs the TFLite model, and returns detections for a single image.

---

## How PlantVillage Works Today (The Upstream System)

PlantVillage is an existing platform at `plantvillage.psu.edu` used by field scouts in Kenyan greenhouses.

### Current Manual Workflow

```
Scout arrives at greenhouse
  → For each Partition (A / B / C):
      → For each Bed (1, 2, 3...):
          → Takes ~16 photos of leaves
          → Manually selects afflictions (Downy Mildew, Mite, Powdery Mildew)
          → Manually sets condition (healthy / existing / new)
  → Submits scout report
  → Admin reviews at /admin/flowers/scouts/{id}
```

### Data Model

```
Greenhouse (id, name, location)
  └── Scout (id, lat, lng, collected_at, user)
        └── Partition (1=A, 2=B, 3=C)
              └── Bed (id, bed_number, problem, sub_affliction)
                    ├── afflictions[]    → {id, name}
                    ├── sub_afflictions[] → {id, name, color}
                    └── photos[]         → {id, url} (S3-hosted .webp)
```

### Known Afflictions

| ID | Name           | Sub-Afflictions          |
|----|----------------|--------------------------|
| 1  | Downy Mildew   | New spots, Spreading ... |
| 2  | Powdery Mildew | New spots, Spreading ... |
| 3  | Mite           | New spots, Spreading ... |
| 4  | Black Spot     | (added by our model)     |

### Upstream API

The PlantVillage admin uses Inertia.js. To get raw JSON, send `X-Inertia: true` header:

| Endpoint | Returns |
|---|---|
| `GET /admin/flowers/scouts` | Paginated scout list |
| `GET /admin/flowers/scouts/{id}` | Scout detail with partitions + beds grid |
| `GET /admin/flowers/scouts/{id}/partitions/{id}` | Beds with photos, afflictions, conditions |
| `GET /admin/flowers/greenhouses` | Greenhouse list with scout counts |

Full reference: [`PLANTVILLAGE_API_REFERENCE.md`](./PLANTVILLAGE_API_REFERENCE.md)

---

## What We Built

### Project Structure

```
Rose_Detection/
├── Training/                    # Model training
│   ├── data/
│   │   ├── black_spot/          # 189 labeled images
│   │   └── downy_mildew/        # labeled images
│   ├── train.py                 # TensorFlow training script
│   ├── export_tflite.py         # Export to TFLite for edge/server
│   └── output/
│       └── labels.txt           # black_spot, downy_mildew
│
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
│
├── Burst_Bed1_Example/          # 5 test burst frames from a real walkby
└── PLANTVILLAGE_API_REFERENCE.md
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

### Local

```bash
cd Backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Interactive Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Testing with Postman

Import `Backend/postman_collection.json` and run the `POST /detect_flowers` request.

---

## What's Next

- [ ] Add `powdery_mildew` and `mite` to training data + retrain model
- [ ] Add S3 photo download to automatically pull bed photos from PlantVillage
- [ ] Edge deployment (TFLite on mobile) for offline detection in the field
