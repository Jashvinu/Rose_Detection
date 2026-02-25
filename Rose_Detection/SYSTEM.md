# PlantVillage Rose Edition — System Documentation

## What We're Building

An automated rose disease detection system for PlantVillage field scouts. Instead of scouts manually identifying diseases on rose leaves, they take burst photos while walking along greenhouse beds, and our system:

1. **Detects** diseased leaves in each photo (SAHI tiling + TFLite classifier)
2. **Tracks** the same leaf across consecutive burst frames (IoU tracker + motion compensation)
3. **Deduplicates** so each physical leaf is counted once, not once per photo
4. **Maps** detections to PlantVillage affliction categories (Downy Mildew, Black Spot, etc.)

The result: a scout walks a bed, takes 16 photos, and gets back "3 leaves with black spot, 1 with downy mildew" — not "47 detections across 16 frames."

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
│       ├── pv_schemas.py        # PlantVillage data models
│       ├── pv_router.py         # PV admin endpoint documentation
│       ├── model/
│       │   ├── rose_disease_model.tflite
│       │   └── labels.txt
│       └── pipeline/
│           ├── classifier.py    # TFLite model wrapper
│           ├── tiler.py         # SAHI sliding-window + NMS + spatial merge
│           ├── tracker.py       # IoU cross-frame tracker
│           ├── motion.py        # ORB + RANSAC motion compensation
│           ├── counter.py       # Count-once with majority-vote labels
│           ├── orchestrator.py  # Wires detect → track → count
│           ├── visualizer.py    # Bounding box annotation on images
│           └── affliction_mapper.py  # Maps labels → PV affliction IDs
│
├── Burst_Bed1_Example/          # 5 test burst frames from a real walkby
└── PLANTVILLAGE_API_REFERENCE.md
```

### Detection Pipeline

```
Input: Ordered burst images from a bed walkby
                    │
                    ▼
        ┌───────────────────┐
        │   SAHI Tiler      │  Slide 224×224 tiles across full-res image
        │   (tiler.py)      │  Classify each tile with TFLite model
        │                   │  NMS + spatial merge overlapping boxes
        └─────────┬─────────┘
                  │ list[Detection] per frame
                  ▼
        ┌───────────────────┐
        │  Motion Comp.     │  ORB feature matching between frames
        │  (motion.py)      │  RANSAC homography estimation
        │                   │  Returns H matrix (prev → current frame)
        └─────────┬─────────┘
                  │ homography matrix (or None on failure)
                  ▼
        ┌───────────────────┐
        │  IoU Tracker      │  Warp previous track boxes through H
        │  (tracker.py)     │  Greedy IoU matching to link same leaf
        │                   │  across frames into persistent Tracks
        └─────────┬─────────┘
                  │ list[Track] (each = one physical leaf)
                  ▼
        ┌───────────────────┐
        │  Counter          │  min_hits confirmation filter
        │  (counter.py)     │  Majority-vote class label per track
        │                   │  Count each confirmed track once
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │  Affliction Map   │  Map model labels → PV affliction IDs
        │  (affliction_     │  Parse scout/partition/bed from token
        │   mapper.py)      │  Derive condition (healthy/new/existing)
        └─────────┬─────────┘
                  │
                  ▼
            JSON Response
```

### How Motion Compensation Works

The problem: a scout walks along a bed taking burst photos. The same leaf appears in frames 1, 2, and 3 but shifted 200px to the left each time. Without compensation, IoU between frames is near zero — the tracker thinks it's 3 different leaves.

The solution:

1. **ORB features** — detect keypoints in each frame (fast, rotation-invariant)
2. **BFMatcher + Lowe's ratio test** — match features between consecutive frames, reject ambiguous matches
3. **RANSAC homography** — fit a perspective transform from prev→current frame, reject outliers
4. **Warp track boxes** — before IoU matching, warp each existing track's last bbox into the current frame's coordinate space
5. **Fallback** — if matching fails (blur, scene change), returns `None` and tracker uses raw IoU (original behavior)

### API Endpoints

#### Detection (our pipeline)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/detect` | Upload images, run pipeline, return results |
| `GET` | `/api/v1/jobs/{job_id}` | Retrieve stored job results |
| `GET` | `/api/v1/jobs/{job_id}/frames/{index}/image` | Annotated frame as JPEG |
| `GET` | `/health` | Server + model status |

#### POST /api/v1/detect — Form Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `images` | File[] | required | Burst photos in sequential order |
| `token` | string | `""` | Session token or PV token (`scout:91\|partition:2\|bed:3338`) |
| `motion_compensation` | bool | `true` | Enable/disable ORB+RANSAC motion compensation |
| `min_hits` | int | `3` | Min frames a leaf must appear in to be counted (use `1` for short bursts) |
| `confidence_threshold` | float | `0.5` | Min classifier confidence |
| `tile_overlap` | float | `0.5` | SAHI tile overlap ratio |
| `visualize` | bool | `true` | Embed annotated images as base64 in response |
| `callback_url` | string | `""` | Webhook URL to POST results to |

#### Response

```json
{
  "token": "scout:91|partition:2|bed:3338",
  "job_id": "abc-123",
  "status": "completed",

  "summary": {
    "black_spot": 3,
    "downy_mildew": 1,
    "total_detections": 4
  },

  "unique_detections": [
    {
      "track_id": 1,
      "label": "black_spot",
      "confidence": 0.92,
      "bbox": {"x1": 120, "y1": 80, "x2": 310, "y2": 260},
      "frame_index": 0,
      "filename": "burst_001.png"
    }
  ],
  "unique_frames": [0, 3],

  "afflictions": [
    {"id": 1, "name": "Downy Mildew"},
    {"id": 4, "name": "Black Spot"}
  ],
  "condition": "new",
  "scout_id": 91,
  "partition_id": 2,
  "bed_id": 3338,

  "per_frame_details": [
    {
      "frame_index": 0,
      "filename": "burst_001.png",
      "detections": [...],
      "active_tracks": 5,
      "annotated_image": "data:image/jpeg;base64,..."
    }
  ],
  "skipped_files": []
}
```

**Key distinction**:
- `per_frame_details[].detections` — every detection in every frame (all get bounding boxes drawn)
- `unique_detections` — one entry per unique physical leaf (deduplicated across frames)
- `summary` — count of unique leaves per disease class

### PlantVillage Admin Endpoints (documented in /docs)

These mirror the upstream PlantVillage system and are available as reference endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/admin/flowers/scouts` | List scouts (paginated) |
| `GET` | `/admin/flowers/scouts/{scout_id}` | Scout detail with partitions + beds |
| `GET` | `/admin/flowers/scouts/{scout_id}/partitions/{partition_id}` | Beds + photos + afflictions |
| `GET` | `/admin/flowers/greenhouses` | List greenhouses |

---

## Configuration

All settings are overridable via `ROSE_` environment variables:

| Setting | Default | Env Var | Description |
|---|---|---|---|
| `tile_size` | 224 | `ROSE_TILE_SIZE` | SAHI tile dimensions (matches model input) |
| `tile_overlap` | 0.5 | `ROSE_TILE_OVERLAP` | Overlap between adjacent tiles |
| `confidence_threshold` | 0.5 | `ROSE_CONFIDENCE_THRESHOLD` | Min confidence to keep a detection |
| `nms_iou_threshold` | 0.45 | `ROSE_NMS_IOU_THRESHOLD` | NMS suppression threshold |
| `match_iou_threshold` | 0.3 | `ROSE_MATCH_IOU_THRESHOLD` | Tracker match threshold |
| `max_age` | 10 | `ROSE_MAX_AGE` | Frames before a lost track is retired |
| `min_hits` | 3 | `ROSE_MIN_HITS` | Min observations to confirm a track |
| `class_agreement_ratio` | 0.7 | `ROSE_CLASS_AGREEMENT_RATIO` | Majority-vote threshold for label |
| `motion_compensation` | true | `ROSE_MOTION_COMPENSATION` | Enable motion compensation |
| `motion_max_features` | 500 | `ROSE_MOTION_MAX_FEATURES` | ORB keypoint limit |
| `motion_min_matches` | 10 | `ROSE_MOTION_MIN_MATCHES` | Min good matches for homography |

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

Import `Backend/postman_collection.json`. Three detection modes:

1. **Basic** — single images, no motion compensation
2. **Burst** — sequential frames with `motion_compensation=true`, `min_hits=1`
3. **PlantVillage Bed Session** — burst + PV token for affliction mapping

---

## What's Next

- [ ] Wire PV admin endpoints to proxy upstream (with session cookie forwarding)
- [ ] Add `powdery_mildew` and `mite` to training data + retrain model
- [ ] Build frontend photo gallery that displays annotated frames with track highlights
- [ ] Add S3 photo download to automatically pull bed photos from PlantVillage
- [ ] Batch processing endpoint for multiple beds in a single scout submission
- [ ] Edge deployment (TFLite on mobile) for offline detection in the field
