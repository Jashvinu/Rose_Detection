"""PlantVillage admin API — documented endpoints.

These endpoints mirror the PlantVillage admin system at plantvillage.psu.edu.
The upstream uses Inertia.js; to get raw JSON, send GET with header X-Inertia: true.

Image storage: AWS S3 (plantvillage-production-new.s3.amazonaws.com)
Auth: Session-based cookie auth (admin pages require login)
"""

from __future__ import annotations

from fastapi import APIRouter, Path, Query

from app.pv_schemas import (
    GreenhousesListResponse,
    PartitionDetailResponse,
    ScoutDetailResponse,
    ScoutsListResponse,
)

router = APIRouter(
    prefix="/admin/flowers",
    tags=["PlantVillage Admin"],
)


PARTITION_LABELS = {1: "A", 2: "B", 3: "C"}


@router.get(
    "/scouts",
    response_model=ScoutsListResponse,
    summary="List all scouts",
    description="""Returns paginated list of scouts with greenhouse, location, and user info.

**Upstream**: `GET /admin/flowers/scouts` (with `X-Inertia: true` header)

**Columns**: Greenhouse | Location | Collected At | Submitted At | User

Also returns global `options.afflictions` (Downy Mildew, Powdery Mildew, Mite)
with their sub-afflictions.""",
)
async def list_scouts(
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=25, ge=1, le=100, description="Results per page"),
):
    # Placeholder — will proxy to upstream or query local DB
    return {
        "scouts": {
            "data": [],
            "meta": {"count": 0, "limit": limit, "page": page},
        },
        "options": {
            "afflictions": [
                {"id": 1, "name": "Downy Mildew", "sub_afflictions": [
                    {"id": 1, "name": "New spots", "color": "#FF6B6B"},
                    {"id": 2, "name": "Spreading", "color": "#FFA500"},
                ]},
                {"id": 2, "name": "Powdery Mildew", "sub_afflictions": [
                    {"id": 3, "name": "New spots", "color": "#FF6B6B"},
                    {"id": 4, "name": "Spreading", "color": "#FFA500"},
                ]},
                {"id": 3, "name": "Mite", "sub_afflictions": [
                    {"id": 5, "name": "New spots", "color": "#FF6B6B"},
                    {"id": 6, "name": "Spreading", "color": "#FFA500"},
                ]},
            ],
        },
    }


@router.get(
    "/scouts/{scout_id}",
    response_model=ScoutDetailResponse,
    summary="Get scout detail",
    description="""Returns full scout with partitions (A/B/C) and beds grid.

**Upstream**: `GET /admin/flowers/scouts/{scout_id}` (with `X-Inertia: true` header)

**Partitions**: 1=A, 2=B, 3=C
**Bed statuses**: `healthy` | `existing` (ongoing issue) | `new` (first sighting)
**Afflictions per bed**: Downy Mildew, Powdery Mildew, Mite""",
)
async def get_scout(
    scout_id: int = Path(description="Scout ID (e.g. 91)"),
):
    return {
        "scout": {
            "id": scout_id,
            "lat": None,
            "lng": None,
            "location": "",
            "collected_at": "",
            "submitted_at": "",
            "updated_at": "",
            "greenhouse": {"id": 0, "name": "", "url": "", "saved_location": None},
            "sections": [],
        },
    }


@router.get(
    "/scouts/{scout_id}/partitions/{partition_id}",
    response_model=PartitionDetailResponse,
    summary="Get partition detail (beds + photos)",
    description="""Returns all beds in a partition with their photos, afflictions, and conditions.

**Upstream**: `GET /admin/flowers/scouts/{scout_id}/partitions/{partition_id}`

Each bed contains:
- `afflictions[]` — structured list of diseases
- `sub_afflictions[]` — severity/type with color coding
- `photos[]` — S3 URLs for all leaf photos
- `problem` — `healthy` | `existing` | `new`

**Photo URL pattern**:
```
https://plantvillage-production-new.s3.amazonaws.com/images%2F{uuid}/flower_{scout_uuid}_{partition}_{bed}_{photo_uuid}.webp
```

**Photo gallery**: ~16 photos per bed, viewable in modal with thumbnail strip.""",
)
async def get_partition(
    scout_id: int = Path(description="Scout ID (e.g. 91)"),
    partition_id: int = Path(description="Partition number: 1=A, 2=B, 3=C"),
):
    return {
        "partition": partition_id,
        "scout_id": scout_id,
        "sections": {"data": []},
    }


@router.get(
    "/greenhouses",
    response_model=GreenhousesListResponse,
    summary="List all greenhouses",
    description="""Returns paginated list of greenhouses with location and scout count.

**Upstream**: `GET /admin/flowers/greenhouses` (with `X-Inertia: true` header)

**Columns**: Name | Location | Scouts Count""",
)
async def list_greenhouses(
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=25, ge=1, le=100, description="Results per page"),
):
    return {
        "greenhouses": {
            "data": [],
            "meta": {"count": 0, "limit": limit, "page": page},
        },
    }
