# PlantVillage Admin — API & Endpoint Reference

> This document maps every page, endpoint, and data payload in the PlantVillage admin system.  
> The system uses **Inertia.js** — data is delivered as JSON in `data-page` HTML attributes, not via separate REST APIs.  
> To get raw JSON from any page, send a GET request with header `X-Inertia: true`.

---

## Architecture

- **Framework**: Inertia.js (server-rendered SPA — likely Laravel/Rails + Vue)
- **Image Storage**: AWS S3 (`plantvillage-production-new.s3.amazonaws.com`)
- **Auth**: Session-based (cookie auth, admin pages require login)
- **Navigation**: Sidebar with Home and Analytics links + breadcrumb trail

---

## Endpoints Overview

| Page | URL | Inertia Component | Method |
|------|-----|-------------------|--------|
| Scouts List | `/admin/flowers/scouts` | `admin/flowers/scouts/index` | GET |
| Scout Detail | `/admin/flowers/scouts/{scout_id}` | `admin/flowers/scouts/show` | GET |
| Partition Detail | `/admin/flowers/scouts/{scout_id}/partitions/{partition_id}` | `admin/flowers/scouts/partitions/show` | GET |
| Greenhouses List | `/admin/flowers/greenhouses` | `admin/flowers/greenhouses/index` | GET |
| Greenhouse Detail | `/admin/flowers/greenhouses/{greenhouse_id}` | `admin/flowers/greenhouses/show` | GET |
| Home Panel | `/panel/home` | (panel page) | GET |
| Analytics | `/panel/analytics/locust_surveys` | (panel page) | GET |

---

## Page 1: Scouts List

**URL**: `GET /admin/flowers/scouts`  
**Component**: `admin/flowers/scouts/index`

### Props

```
scouts
├── data[]                          # Array of Scout summary objects
│   ├── id: int                     # Scout ID (e.g. 91)
│   ├── collected_at: string        # "18 February 2026"
│   ├── collected_at_timezone: str  # "Africa/Nairobi"
│   ├── created_at: string          # "18 February 2026"
│   ├── submitted_at: string        # "18 February 2026"
│   ├── location: string            # "Kenya, Nakuru"
│   ├── removable: bool             # Whether scout can be deleted
│   ├── greenhouse                  # Nested greenhouse ref
│   │   ├── id: int
│   │   └── name: string
│   ├── saved_location              # Nested location ref
│   │   ├── id: int
│   │   └── name: string
│   └── user                        # Scout who collected data
│       ├── id: int
│       └── name: string
├── meta                            # Pagination
│   ├── count: int                  # Total scouts
│   ├── limit: int                  # Per page
│   └── page: int                   # Current page
options
├── afflictions[]                   # Global affliction definitions
│   ├── id: int
│   ├── name: string                # "Downy Mildew", "Mite", "Powdery Mildew"
│   └── sub_afflictions[]           # Sub-categories
│       ├── id: int
│       ├── name: string            # "New spots", "Spreading", etc.
│       └── color: string           # Hex or named color
```

### UI
- Table with columns: Greenhouse, Location, Collected At, Submitted At, User
- Each row links to Scout Detail page
- Pagination controls at bottom

---

## Page 2: Scout Detail

**URL**: `GET /admin/flowers/scouts/{scout_id}`  
**Component**: `admin/flowers/scouts/show`

### Props

```
scout
├── id: int                         # 91
├── lat: float                      # Latitude
├── lng: float                      # Longitude
├── location: string                # "Kenya, Nakuru"
├── collected_at: string            # Formatted date
├── submitted_at: string
├── updated_at: string
├── greenhouse
│   ├── id: int
│   ├── name: string
│   ├── url: string                 # "/admin/flowers/greenhouses/108"
│   └── saved_location
│       ├── id: int
│       └── name: string
├── sections[]                      # ← PARTITIONS
│   ├── partition: int              # 1, 2, 3 (maps to A, B, C)
│   └── sections[]                  # ← BEDS within this partition
│       ├── id: int                 # Bed/Section ID (e.g. 3338)
│       ├── bed_number: int         # 1, 2, 3...
│       ├── problem: string         # "existing", "healthy", "new"
│       └── afflictions[]           # Diseases detected in this bed
│           ├── id: int
│           └── name: string
```

### UI
- **Partitions & Beds Grid**: rows = Partition A/B/C, columns = Bed numbers
- **Disease filter tabs** at top: Downy Mildew | Mite | Powdery Mildew
- **Status icons**: ⚠️ (existing issue), ✅ (healthy), 🔴 (new issue)
- Clicking a partition row navigates to Partition Detail

### URL Mapping
- Partition number 1 → "Partition A", 2 → "Partition B", 3 → "Partition C"
- Grid icon click → `GET /admin/flowers/scouts/{scout_id}/partitions/{partition_number}`

---

## Page 3: Partition Detail

**URL**: `GET /admin/flowers/scouts/{scout_id}/partitions/{partition_id}`  
**Component**: `admin/flowers/scouts/partitions/show`

### Props

```
partition: int                      # Partition number (1, 2, 3)
scout_id: int                       # Parent scout ID
sections
├── data[]                          # Array of Bed objects
│   ├── id: int                     # Bed/Section ID (e.g. 3338)
│   ├── bed_number: int
│   ├── affliction: string          # Joined string: "Downy Mildew, Powdery Mildew, Mite"
│   ├── afflictions[]               # Structured array
│   │   ├── id: int
│   │   └── name: string
│   ├── problem: string             # "existing" | "healthy" | "new"
│   ├── sub_affliction: string      # "New spots"
│   ├── sub_afflictions[]           # Structured array
│   │   ├── id: int
│   │   ├── name: string
│   │   └── color: string
│   └── photos[]                    # All photos for this bed
│       ├── id: int
│       └── url: string             # Full S3 URL
```

### UI
- **Breadcrumb**: Flowers > Scouts > Scout > Partition B
- **Table**: Bed | Condition | Photos | Problem | Sub Problem
- **Condition column**: icon + text (e.g. ⚠️ "Existing issue")
- **Photos column**: count (e.g. "16")
- Clicking a bed row opens the **Photo Gallery Modal**

### Photo Gallery Modal
- Large photo viewer with thumbnail strip on left
- Header: "Bed 1, partition B"
- Shows afflictions and sub-affliction
- Photo counter: "Photo 1/16"
- Navigate between photos with clicks or arrows

### Image URL Pattern
```
https://plantvillage-production-new.s3.amazonaws.com/images%2F{uuid}/flower_{scout_uuid}_{partition}_{bed}_{photo_uuid}.webp
```

---

## Page 4: Greenhouses List

**URL**: `GET /admin/flowers/greenhouses`  
**Component**: `admin/flowers/greenhouses/index`

### Props

```
greenhouses
├── data[]                          # Array of Greenhouse objects
│   ├── id: int
│   ├── name: string
│   ├── saved_location
│   │   ├── id: int
│   │   └── name: string
│   └── scouts_count: int           # Number of scouts for this greenhouse
├── meta                            # Pagination
│   ├── count: int
│   ├── limit: int
│   └── page: int
```

### UI
- Table: Name, Location, Scouts Count
- Each row links to Greenhouse Detail or filtered Scouts List

---

## Page 5: Home Panel

**URL**: `GET /panel/home`  
**Component**: Panel home page (different section from flowers admin)

### Navigation Sidebar
- **Home** → `/panel/home`
- **Analytics** → `/panel/analytics/locust_surveys`

> Note: The panel pages are separate from the flowers admin. The flowers admin has its own sidebar with Home and Analytics links.

---

## Global Data: Afflictions

These are the known affliction types defined system-wide:

| ID | Name | Sub-Afflictions |
|----|------|----------------|
| 1 | Downy Mildew | New spots, Spreading, ... |
| 2 | Powdery Mildew | New spots, Spreading, ... |
| 3 | Mite | New spots, Spreading, ... |

Sub-afflictions have:
- `id`: int
- `name`: string (e.g. "New spots")
- `color`: string (visual indicator color)

---

## Data Flow: How PlantVillage Works

```
1. Scout goes to greenhouse with phone
2. Opens PlantVillage app
3. For each Partition (A/B/C):
     For each Bed (1, 2, 3...):
       Takes ~16 photos of leaves
       Selects afflictions seen (Downy Mildew, Mite, etc.)
       Sets condition (healthy / existing / new)
       Sets sub-affliction (New spots / Spreading)
4. Submits scout → uploaded to server
5. Admin views at /admin/flowers/scouts/{id}
```

---

## How to Access Raw JSON

For any PlantVillage admin page, send a GET request with the Inertia header:

```bash
curl -H "X-Inertia: true" \
     -H "X-Inertia-Version: {version_hash}" \
     -H "Cookie: {your_session_cookie}" \
     https://plantvillage.psu.edu/admin/flowers/scouts/91
```

Response format:
```json
{
  "component": "admin/flowers/scouts/show",
  "props": {
    "scout": { ... }
  },
  "url": "/admin/flowers/scouts/91",
  "version": "abc123..."
}
```

---

## Relationship Diagram

```
Greenhouse (id, name, location)
    └── Scout (id, lat, lng, collected_at, user)
          └── Partition (1=A, 2=B, 3=C)
                └── Bed/Section (id, bed_number, problem, sub_affliction)
                      ├── afflictions[] (id, name)
                      ├── sub_afflictions[] (id, name, color)
                      └── photos[] (id, url → S3)
```
