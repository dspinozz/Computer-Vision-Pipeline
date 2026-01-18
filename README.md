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
python scripts/download_places365.py --output-dir data/places365 --use-validation

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

---

## 🚀 Deployment Options

### Option A: Local Development (Recommended for Processing)

Best for: Running the GPU processing pipeline, development, debugging.

#### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for Postgres/Qdrant)
- NVIDIA GPU with CUDA (16GB+ VRAM recommended for full pipeline)
- ~20GB disk space for models

> **GPU Memory Note**: Running `--lane all` requires ~16GB VRAM. For 12GB GPUs, run lanes separately (Lane A, then Lane B). For 8GB GPUs, consider CPU-only mode or cloud GPU.

#### Steps

```bash
# 1. Clone and setup
git clone https://github.com/dspinozz/Computer-Vision-Pipeline.git
cd Computer-Vision-Pipeline

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install sentencepiece  # Required for SigLIP tokenizer

# 2. Download models (~16GB)
python scripts/download_models.py

# 3. Start infrastructure (Postgres + Qdrant)
docker compose up -d postgres qdrant

# 4. Download sample data
python scripts/download_places365.py --output-dir data/places365 --use-validation

# 5. Create manifest (catalog images)
python scripts/pipeline.py ingest --input-dir data/places365/val_256

# 6. Process images (requires GPU)
# Option A: Full pipeline (16GB+ VRAM)
python scripts/pipeline.py process --lane all

# Option B: Separate lanes (12GB VRAM)
python scripts/pipeline.py process --lane A  # Embeddings, captions, tags
python scripts/pipeline.py process --lane B  # Object detection, masks

# 7. Import to databases
python scripts/import_qdrant.py
python scripts/migrate_to_postgres.py

# 8. Start API locally
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

### Option B: Full Docker Deployment (Recommended for Serving)

Best for: Production deployment, serving pre-processed data.

#### Prerequisites

- Docker & Docker Compose
- Pre-processed outputs (metadata JSON files + embeddings)

#### Steps

```bash
# 1. Clone repository
git clone https://github.com/dspinozz/Computer-Vision-Pipeline.git
cd Computer-Vision-Pipeline

# 2. Configure environment (optional, uses defaults otherwise)
cp .env.example .env
# Edit .env to set POSTGRES_PASSWORD, etc.

# 3. Start all services
docker compose up -d postgres qdrant api

# 4. Verify services are running
docker compose ps

# 5. Check API health
curl http://localhost:8000/health
```

#### With GPU (for on-demand processing)

```bash
# Start with GPU-enabled API
docker compose --profile gpu up -d postgres qdrant api-gpu
```

---

## 🧪 Testing the Deployment

### 1. Check Service Health

```bash
# API health
curl http://localhost:8000/health

# Qdrant health
curl http://localhost:6333/health

# Postgres connection (from container)
docker exec cvpipeline-postgres pg_isready -U cvpipeline
```

### 2. Test Semantic Search

```bash
# Search for images
curl "http://localhost:8000/v1/images/search?q=mountain+landscape&limit=3"

# Expected: JSON with matching images, scores, captions
```

### 3. Test Stats Endpoint

```bash
curl http://localhost:8000/v1/stats

# Expected: {"metadata_files": 36497, "embedding_files": 36497, ...}
```

### 4. Run Demo Script

```bash
python notebooks/demo.py
```

### 5. Interactive Testing (Optional)

```bash
# Start pgAdmin for database inspection
docker compose --profile admin up -d pgadmin

# Access at http://localhost:5050
# Login: admin@localhost / admin (change in production)
```

---

## 📁 Project Structure

```
Computer-Vision-Pipeline/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI application
├── configs/
│   ├── schema.sql           # PostgreSQL schema
│   ├── taxonomy.json        # Tag taxonomy (155 tags)
│   └── image_spec.json      # Processing configuration
├── scripts/
│   ├── pipeline.py          # Main processing pipeline
│   ├── download_places365.py # Data download script
│   ├── download_models.py   # Model download script
│   ├── import_qdrant.py     # Vector database import
│   ├── migrate_to_postgres.py # Metadata migration
│   └── test_models.py       # Model testing
├── notebooks/
│   └── demo.py              # Demo script
├── docker-compose.yml       # Multi-service orchestration
├── Dockerfile.api           # API container
├── .env.example             # Environment template
├── requirements.txt         # Full dependencies
├── requirements-minimal.txt # Serving-only dependencies
└── requirements-api.txt     # API dependencies
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
2. Set secure passwords for `POSTGRES_PASSWORD`, `PGADMIN_PASSWORD`
3. Use `docker compose --env-file .env up`

## License

MIT

---

## ⚠️ Model Setup (Required Before Processing)

The pipeline requires pre-downloaded models. Run this **before** processing:

```bash
# Download models (~10GB total, one-time setup)
python scripts/download_models.py

# Or download from HuggingFace manually:
# - google/siglip-so400m-patch14-384 → models/siglip-so400m
# - Qwen/Qwen2.5-VL-3B-Instruct → models/qwen2.5-vl-3b
# - IDEA-Research/grounding-dino-base → models/grounding-dino-base
# - facebook/sam2-hiera-tiny → models/sam2-tiny
```

### Alternative: Copy models from another machine

```bash
scp -r source-machine:/data/image_pipeline/models/* /data/image_pipeline/models/
```
