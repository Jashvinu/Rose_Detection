"""Pydantic models mirroring the PlantVillage admin API data structures."""

from __future__ import annotations

from pydantic import BaseModel


# ── Shared / nested models ───────────────────────────────────────────────────

class NamedRef(BaseModel):
    id: int
    name: str


class SubAffliction(BaseModel):
    id: int
    name: str
    color: str


class PVAffliction(BaseModel):
    id: int
    name: str
    sub_afflictions: list[SubAffliction] = []


class PaginationMeta(BaseModel):
    count: int
    limit: int
    page: int


# ── Scouts List ──────────────────────────────────────────────────────────────

class ScoutSummary(BaseModel):
    """One row in the scouts list table."""
    id: int
    collected_at: str
    collected_at_timezone: str = "Africa/Nairobi"
    created_at: str
    submitted_at: str
    location: str
    removable: bool = False
    greenhouse: NamedRef
    saved_location: NamedRef
    user: NamedRef


class ScoutsListResponse(BaseModel):
    """GET /admin/flowers/scouts"""
    scouts: ScoutsData
    options: ScoutsOptions


class ScoutsData(BaseModel):
    data: list[ScoutSummary]
    meta: PaginationMeta


class ScoutsOptions(BaseModel):
    afflictions: list[PVAffliction]


# Fix forward ref
ScoutsListResponse.model_rebuild()


# ── Scout Detail ─────────────────────────────────────────────────────────────

class BedAffliction(BaseModel):
    id: int
    name: str


class BedSummary(BaseModel):
    """A bed within a partition (scout detail view)."""
    id: int
    bed_number: int
    problem: str
    afflictions: list[BedAffliction] = []


class PartitionSection(BaseModel):
    """A partition row in the scout detail grid."""
    partition: int
    sections: list[BedSummary] = []


class GreenhouseRef(BaseModel):
    id: int
    name: str
    url: str = ""
    saved_location: NamedRef | None = None


class ScoutDetail(BaseModel):
    """Full scout object returned on the detail page."""
    id: int
    lat: float | None = None
    lng: float | None = None
    location: str
    collected_at: str
    submitted_at: str
    updated_at: str = ""
    greenhouse: GreenhouseRef
    sections: list[PartitionSection] = []


class ScoutDetailResponse(BaseModel):
    """GET /admin/flowers/scouts/{scout_id}"""
    scout: ScoutDetail


# ── Partition Detail ─────────────────────────────────────────────────────────

class Photo(BaseModel):
    id: int
    url: str


class BedDetail(BaseModel):
    """Full bed object on the partition detail page."""
    id: int
    bed_number: int
    affliction: str = ""
    afflictions: list[BedAffliction] = []
    problem: str
    sub_affliction: str = ""
    sub_afflictions: list[SubAffliction] = []
    photos: list[Photo] = []


class PartitionSections(BaseModel):
    data: list[BedDetail]


class PartitionDetailResponse(BaseModel):
    """GET /admin/flowers/scouts/{scout_id}/partitions/{partition_id}"""
    partition: int
    scout_id: int
    sections: PartitionSections


# ── Greenhouses List ─────────────────────────────────────────────────────────

class GreenhouseSummary(BaseModel):
    id: int
    name: str
    saved_location: NamedRef | None = None
    scouts_count: int = 0


class GreenhousesData(BaseModel):
    data: list[GreenhouseSummary]
    meta: PaginationMeta


class GreenhousesListResponse(BaseModel):
    """GET /admin/flowers/greenhouses"""
    greenhouses: GreenhousesData
