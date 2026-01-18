# Computer Vision Pipeline

[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

A production-grade vision processing pipeline that extracts embeddings, generates captions, detects objects, and creates pixel masks for large image collections. Supports semantic search via Qdrant and structured queries via PostgreSQL.

## 🎯 Features

- **Semantic Search**: Find images by natural language queries
- **Object Detection**: Identify and localize objects with bounding boxes
- **Mask Generation**: Pixel-precise segmentation for interactive regions
- **Metadata Extraction**: Rich captions, tags, and scene descriptions
- **Production Infrastructure**: FastAPI, Qdrant, PostgreSQL, Docker

## 📊 Pipeline Results

Processed **36,497 images** from Places365 dataset:

| Component | Model | Output |
|-----------|-------|--------|
| Embeddings | SigLIP So400m (1152-dim) | 36,497 vectors |
| Captions | Qwen2.5-VL-3B-Instruct | Avg 199 chars/image |
| Tags | SigLIP + Taxonomy (155 tags) | 10 ranked tags/image |
| Object Detection | GroundingDINO | 77,344 objects |
| Masks | SAM2 | 77,334 pixel masks |

## 📁 Data Source

This pipeline was developed and tested using the **Places365 validation set**:

- **Dataset**: [Places365-Standard](http://places2.csail.mit.edu/download.html)
- **Split**: val_256 (validation images, 256x256)
- **Size**: 36,500 images across 365 scene categories
- **License**: Creative Commons Attribution (CC BY)

To recreate:

```bash
# Download Places365 validation set
python scripts/download_places365.py --output-dir data/places365

# Or manually download from:
# http://data.csail.mit.edu/places/places365/val_256.tar
```

The pipeline works with any image collection - just point it to your image directory.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Input Images                                │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Lane A: Metadata                              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  SigLIP  │  │ Qwen2.5-VL   │  │   SigLIP     │               │
│  │Embedding │  │  Captioning  │  │ Tag Scoring  │               │
│  └────┬─────┘  └──────┬───────┘  └──────┬───────┘               │
│       │               │                 │                        │
│       ▼               ▼                 ▼                        │
│   1152-dim        Caption +         10 ranked                   │
│    vector        short + nouns        tags                       │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────────────┐
│                   Lane B: Detection                              │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │ GroundingDINO│         │     SAM2     │                      │
│  │  Detection   │────────▶│    Masks     │                      │
│  └──────────────┘         └──────────────┘                      │
│         │                        │                               │
│         ▼                        ▼                               │
│   Bounding boxes          Pixel masks +                         │
│   + labels                 centroids                             │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐               │
│  │  Qdrant  │    │ Postgres │    │  JSON Files  │               │
│  │ Vectors  │    │ Metadata │    │   (backup)   │               │
│  └──────────┘    └──────────┘    └──────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI                                     │
│  GET /v1/images/search?q=...    Semantic search                 │
│  GET /v1/images/{id}            Image details                   │
│  GET /v1/stats                  Collection stats                │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- NVIDIA GPU (for processing; serving is CPU-only)

### 1. Clone and Setup

```bash
git clone https://github.com/dspinozz/Computer-Vision-Pipeline.git
cd Computer-Vision-Pipeline

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download Places365 validation set (~1GB)
python scripts/download_places365.py --output-dir data/places365
```

### 3. Start Infrastructure

```bash
docker compose up -d postgres qdrant
```

### 4. Process Images

```bash
# Run full pipeline (requires GPU)
python scripts/pipeline.py process --input-dir data/places365/val_256

# Import to vector database
python scripts/import_qdrant.py

# Migrate to PostgreSQL
python scripts/migrate_to_postgres.py
```

### 5. Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 6. Search

```bash
curl "http://localhost:8000/v1/images/search?q=mountain+landscape"
```

## 📁 Project Structure

```
Computer-Vision-Pipeline/
├── api/
│   └── main.py              # FastAPI application
├── configs/
│   ├── schema.sql           # PostgreSQL schema
│   └── taxonomy.json        # Tag taxonomy (155 tags)
├── scripts/
│   ├── pipeline.py          # Main processing pipeline
│   ├── download_places365.py # Data download script
│   ├── import_qdrant.py     # Vector import
│   └── migrate_to_postgres.py
├── notebooks/
│   └── demo.py              # Demo script
├── docker-compose.yml
├── Dockerfile.api
├── .env.example             # Environment template
└── requirements.txt
```

## 🔧 Models Used

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| SigLIP So400m | Embeddings + tags | 400M | [HuggingFace](https://huggingface.co/google/siglip-so400m-patch14-384) |
| Qwen2.5-VL-3B | Captioning | 3B | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| GroundingDINO | Object detection | 172M | [GitHub](https://github.com/IDEA-Research/GroundingDINO) |
| SAM2 | Segmentation | 312M | [GitHub](https://github.com/facebookresearch/sam2) |

## 📊 Metadata Schema

Each image produces structured metadata:

```json
{
  "image_id": "uuid",
  "file_path": "data/places365/val_256/image.jpg",
  "file_hash": "sha256...",
  "width": 256,
  "height": 256,
  "caption": {
    "text": "Full VLM-generated description...",
    "short": "First 100 chars...",
    "extracted_nouns": ["mountain", "sky", "tree"]
  },
  "tags": [
    {"tag_id": "scene.mountain", "display": "Mountain", "confidence": 0.14}
  ],
  "objects": [
    {
      "label": "person",
      "confidence": 0.87,
      "box": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.8},
      "mask_rle": "...",
      "mask_area": 1024,
      "centroid": [128.5, 192.3]
    }
  ]
}
```

## 🎯 Use Cases

- **Semantic Image Search**: Find images by natural language description
- **Interactive Applications**: Clickable regions with pixel-precise masks
- **Content Moderation**: Detect and classify image content
- **Accessibility**: Generate alt-text and scene descriptions
- **Asset Libraries**: Search stock images by semantic similarity

## 📈 Performance

- **Processing speed**: ~2 images/sec (GPU)
- **Qdrant search**: <50ms for 36K vectors
- **API latency**: <100ms (cached embeddings)

## 🔒 Security Note

Default passwords in `docker-compose.yml` are for **development only**. 
For production:
1. Copy `.env.example` to `.env`
2. Set secure passwords
3. Use `docker compose --env-file .env up`

## License

MIT
