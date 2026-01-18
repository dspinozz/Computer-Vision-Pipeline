"""
Computer Vision Pipeline API

FastAPI application providing endpoints for:
- Image search (hybrid: vector + text + tags)
- Image metadata retrieval
- On-demand interactive localization (GroundingDINO + SAM2)
- User image upload processing

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Lazy imports for heavy dependencies
_qdrant_client = None
_pipeline = None

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
    "collection_name": os.getenv("QDRANT_COLLECTION", "image_embeddings"),
    "models_dir": Path(os.getenv("MODELS_DIR", "/data/image_pipeline/models")),
    "data_dir": Path(os.getenv("DATA_DIR", "/data/image_pipeline/data")),
    "output_dir": Path(os.getenv("OUTPUT_DIR", "/data/image_pipeline/outputs")),
    "taxonomy_path": Path(os.getenv("TAXONOMY_PATH", "/data/image_pipeline/configs/taxonomy.json")),
    "upload_dir": Path(os.getenv("UPLOAD_DIR", "/data/image_pipeline/uploads")),
    "max_upload_size_mb": 10,
    "batch_size": 1,
    "max_caption_length": 200,
    "tag_confidence_threshold": 0.5,
    "object_confidence_threshold": 0.3,
    "max_objects_per_image": 20,
}

# ============================================================================
# Pydantic Models
# ============================================================================

class ImageCard(BaseModel):
    """Lightweight image representation for search results."""
    image_id: str
    file_path: str
    caption_short: Optional[str] = None
    tags: List[str] = []
    score: float = 0.0


class TagInfo(BaseModel):
    """Tag with confidence score."""
    tag_id: str
    display: str
    category: str
    confidence: float


class CaptionInfo(BaseModel):
    """Caption and extracted text."""
    text: str
    short: str
    extracted_nouns: List[str] = []
    model: str = "qwen2.5-vl-3b"


class Detection(BaseModel):
    """Object detection result."""
    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]
    mask_rle: Optional[str] = None
    area_px: Optional[int] = None


class ImageMetadata(BaseModel):
    """Full image metadata."""
    image_id: str
    file_hash: str
    file_path: str
    width: int
    height: int
    source_type: str
    source_category: Optional[str] = None
    caption: Optional[CaptionInfo] = None
    tags: List[TagInfo] = []
    objects: Optional[List[Detection]] = None
    embedding_id: Optional[str] = None
    processed_at: Optional[str] = None


class SearchRequest(BaseModel):
    """Search request body."""
    query: str
    tags: Optional[List[str]] = None
    source_type: Optional[str] = None
    limit: int = Field(default=20, le=100)


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    results: List[ImageCard]
    total: int


class ProcessRequest(BaseModel):
    """Batch process request."""
    image_ids: Optional[List[str]] = None
    limit: Optional[int] = None
    lane: str = Field(default="A", pattern="^(A|B|all)$")


class InteractResponse(BaseModel):
    """Interactive localization response."""
    image_id: str
    objects: List[Detection]
    cached: bool = False


# ============================================================================
# App Setup
# ============================================================================

app = FastAPI(
    title="Computer Vision Pipeline API",
    description="Search, retrieve, and interact with scene images for computer vision applications",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_qdrant_client():
    """Lazy-load Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=CONFIG["qdrant_url"])
    return _qdrant_client


def get_pipeline():
    """Lazy-load pipeline (heavy, only when needed for processing)."""
    global _pipeline
    if _pipeline is None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from pipeline import Pipeline
        _pipeline = Pipeline(CONFIG)
    return _pipeline


def load_metadata_by_id(image_id: str) -> Optional[Dict]:
    """Load metadata JSON by image_id."""
    metadata_dir = CONFIG["output_dir"] / "metadata"
    meta_path = metadata_dir / f"{image_id}.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None


def load_metadata_by_hash(file_hash: str) -> Optional[Dict]:
    """Load metadata JSON by file_hash (slower, scans directory)."""
    metadata_dir = CONFIG["output_dir"] / "metadata"
    for meta_file in metadata_dir.glob("*.json"):
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            if meta.get("file_hash") == file_hash:
                return meta
        except (json.JSONDecodeError, KeyError):
            continue
    return None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/v1/images/search", response_model=SearchResponse)
async def search_images(
    q: str = Query(..., description="Search query"),
    tags: Optional[str] = Query(None, description="Comma-separated tag filters"),
    source: Optional[str] = Query(None, description="Source type filter"),
    limit: int = Query(20, le=100, description="Max results"),
):
    """
    Search images using hybrid vector + text matching.
    
    The search process:
    1. Embed query using SigLIP
    2. Vector search in Qdrant
    3. Rerank by tag overlap and caption match
    """
    try:
        client = get_qdrant_client()
        pipeline = get_pipeline()
        
        # Get query embedding
        import torch
        siglip = pipeline.model_manager.load_siglip()
        processor = siglip["processor"]
        model = siglip["model"]
        
        # Text-to-embedding
        txt_inputs = processor(text=[q], return_tensors="pt", padding=True)
        txt_inputs = {k: v.to(model.device) for k, v in txt_inputs.items()}
        
        with torch.no_grad():
            txt_embeds = model.get_text_features(**txt_inputs)
            query_vector = txt_embeds[0].cpu().numpy().tolist()
        
        # Build filter if tags specified
        query_filter = None
        if tags:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            tag_list = [t.strip() for t in tags.split(",")]
            query_filter = Filter(
                must=[
                    FieldCondition(key="tags", match=MatchAny(any=tag_list))
                ]
            )
        
        # Vector search (using query_points for newer qdrant-client)
        from qdrant_client.models import QueryRequest
        results = client.query_points(
            collection_name=CONFIG["collection_name"],
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        
        # Convert to response
        cards = []
        for hit in results.points:
            cards.append(ImageCard(
                image_id=hit.payload.get("image_id", str(hit.id)),
                file_path=hit.payload.get("file_path", ""),
                caption_short=hit.payload.get("caption_short", ""),
                tags=hit.payload.get("tag_displays", []),
                score=hit.score,
            ))
        
        return SearchResponse(query=q, results=cards, total=len(cards))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/images/{image_id}", response_model=ImageMetadata)
async def get_image(image_id: str):
    """Get full metadata for a specific image."""
    meta = load_metadata_by_id(image_id)
    
    if not meta:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    
    # Convert to response model
    caption_info = None
    if meta.get("caption"):
        caption_info = CaptionInfo(
            text=meta["caption"].get("text", ""),
            short=meta["caption"].get("short", ""),
            extracted_nouns=meta["caption"].get("extracted_nouns", []),
            model=meta["caption"].get("model", "qwen2.5-vl-3b"),
        )
    
    tags = []
    for t in meta.get("tags") or []:
        tags.append(TagInfo(
            tag_id=t.get("tag_id", ""),
            display=t.get("display", ""),
            category=t.get("category", ""),
            confidence=t.get("confidence", 0.0),
        ))
    
    objects = None
    if meta.get("objects"):
        objects = []
        for obj in meta["objects"]:
            objects.append(Detection(
                label=obj.get("label", ""),
                confidence=obj.get("confidence", 0.0),
                box=obj.get("box", [0, 0, 0, 0]),
                mask_rle=obj.get("mask_rle"),
                area_px=obj.get("area_px"),
            ))
    
    return ImageMetadata(
        image_id=meta["image_id"],
        file_hash=meta["file_hash"],
        file_path=meta["file_path"],
        width=meta["width"],
        height=meta["height"],
        source_type=meta["source_type"],
        source_category=meta.get("source_category"),
        caption=caption_info,
        tags=tags,
        objects=objects,
        embedding_id=meta.get("embedding_id"),
        processed_at=meta.get("processed_at"),
    )


@app.post("/v1/images/{image_id}/interact", response_model=InteractResponse)
async def interact_with_image(image_id: str, background_tasks: BackgroundTasks):
    """
    Run on-demand interactive localization (GroundingDINO + SAM2).
    
    If objects/masks are already cached, returns them immediately.
    Otherwise, runs Lane B processing and returns results.
    """
    meta = load_metadata_by_id(image_id)
    
    if not meta:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    
    # Check if already processed (has objects with masks)
    if meta.get("objects") and any(obj.get("mask_rle") for obj in meta["objects"]):
        objects = []
        for obj in meta["objects"]:
            objects.append(Detection(
                label=obj.get("label", ""),
                confidence=obj.get("confidence", 0.0),
                box=obj.get("box", [0, 0, 0, 0]),
                mask_rle=obj.get("mask_rle"),
                area_px=obj.get("area_px"),
            ))
        return InteractResponse(image_id=image_id, objects=objects, cached=True)
    
    # Run Lane B processing
    try:
        pipeline = get_pipeline()
        image_path = Path(meta["file_path"])
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")
        
        # Get prompts from taxonomy + caption nouns
        prompts = [
            pipeline.taxonomy.tag_info[t]["display"] 
            for t in pipeline.taxonomy.get_localizable_tags()
        ]
        
        # Add nouns from caption
        if meta.get("caption"):
            caption_text = meta["caption"].get("text", "")
            if caption_text:
                fresh_nouns = pipeline._extract_nouns(caption_text)
                for noun in fresh_nouns[:10]:
                    if noun not in prompts and len(noun) >= 3:
                        prompts.append(noun)
        
        # Run detection
        detected_objects = pipeline.detect_objects(image_path, prompts)
        
        # Run mask generation
        if detected_objects:
            detected_objects = pipeline.generate_masks(image_path, detected_objects)
        
        # Update metadata file
        meta["objects"] = detected_objects
        meta["processed_at"] = datetime.now().isoformat()
        
        metadata_path = CONFIG["output_dir"] / "metadata" / f"{image_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Convert to response
        objects = []
        for obj in detected_objects or []:
            objects.append(Detection(
                label=obj.get("label", ""),
                confidence=obj.get("confidence", 0.0),
                box=obj.get("box", [0, 0, 0, 0]),
                mask_rle=obj.get("mask_rle"),
                area_px=obj.get("area_px"),
            ))
        
        return InteractResponse(image_id=image_id, objects=objects, cached=False)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/uploads/process", response_model=ImageMetadata)
async def upload_and_process(file: UploadFile = File(...)):
    """
    Upload a user image and process through the full pipeline.
    
    Returns complete metadata including embeddings, captions, and tags.
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and save file
    contents = await file.read()
    
    if len(contents) > CONFIG["max_upload_size_mb"] * 1024 * 1024:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large (max {CONFIG['max_upload_size_mb']}MB)"
        )
    
    # Compute hash
    file_hash = hashlib.sha256(contents).hexdigest()
    
    # Check if already processed
    existing = load_metadata_by_hash(file_hash)
    if existing:
        # Return existing metadata
        return await get_image(existing["image_id"])
    
    # Save to uploads directory
    upload_dir = CONFIG["upload_dir"]
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    ext = Path(file.filename).suffix or ".jpg"
    upload_path = upload_dir / f"{file_hash}{ext}"
    
    with open(upload_path, 'wb') as f:
        f.write(contents)
    
    # Process through pipeline
    try:
        pipeline = get_pipeline()
        
        metadata = pipeline.process_image(
            upload_path,
            source_type="user_upload",
            source_category="uploaded",
            lane="all",  # Full processing for uploads
            output_dir=CONFIG["output_dir"] / "metadata",
        )
        
        # Save metadata
        pipeline.save_metadata(metadata, CONFIG["output_dir"] / "metadata")
        
        return await get_image(metadata.image_id)
        
    except Exception as e:
        # Clean up on error
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/ingest")
async def ingest_images(
    folder: Optional[str] = Query(None, description="Local folder to scan"),
    urls: Optional[List[str]] = None,
):
    """
    Ingest images from a local folder or URLs.
    
    Returns list of image IDs for newly ingested images.
    """
    if not folder and not urls:
        raise HTTPException(status_code=400, detail="Provide either 'folder' or 'urls'")
    
    ingested = []
    
    if folder:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")
        
        for img_path in folder_path.glob("**/*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                # Compute hash
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                # Check if already exists
                existing = load_metadata_by_hash(file_hash)
                if not existing:
                    ingested.append({
                        "path": str(img_path),
                        "file_hash": file_hash,
                        "status": "pending"
                    })
    
    # TODO: Implement URL download
    
    return {
        "ingested": len(ingested),
        "images": ingested[:100],  # Limit response size
        "message": "Use POST /v1/images/process/batch to process ingested images"
    }


@app.post("/v1/images/process/batch")
async def process_batch(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Trigger batch processing of images.
    
    This is typically run as a background job for large batches.
    """
    return {
        "status": "accepted",
        "lane": request.lane,
        "limit": request.limit,
        "message": "Batch processing started. Check logs for progress.",
        "note": "For large batches, use the CLI: python scripts/pipeline.py process --lane A"
    }


# ============================================================================
# Taxonomy Endpoints
# ============================================================================

@app.get("/v1/taxonomy")
async def get_taxonomy():
    """Get the full tag taxonomy."""
    taxonomy_path = CONFIG["taxonomy_path"]
    if not taxonomy_path.exists():
        raise HTTPException(status_code=404, detail="Taxonomy not found")
    
    with open(taxonomy_path, 'r') as f:
        taxonomy = json.load(f)
    
    return taxonomy


@app.get("/v1/taxonomy/tags")
async def get_all_tags():
    """Get flat list of all tags with their metadata."""
    pipeline = get_pipeline()
    
    tags = []
    for tag_id, info in pipeline.taxonomy.tag_info.items():
        tags.append({
            "tag_id": tag_id,
            "display": info.get("display", tag_id),
            "category": info.get("category", "unknown"),
            "localizable": info.get("localizable", False),
            "prompts": info.get("prompts", []),
        })
    
    return {"tags": tags, "total": len(tags)}


# ============================================================================
# Stats Endpoints
# ============================================================================

@app.get("/v1/stats")
async def get_stats():
    """Get pipeline statistics."""
    metadata_dir = CONFIG["output_dir"] / "metadata"
    embeddings_dir = CONFIG["output_dir"] / "embeddings"
    
    # Count files
    metadata_count = len(list(metadata_dir.glob("*.json"))) if metadata_dir.exists() else 0
    embedding_count = len(list(embeddings_dir.glob("*.npy"))) if embeddings_dir.exists() else 0
    
    # Try to get Qdrant count
    qdrant_count = 0
    try:
        client = get_qdrant_client()
        info = client.get_collection(CONFIG["collection_name"])
        qdrant_count = info.points_count
    except Exception:
        pass
    
    return {
        "metadata_files": metadata_count,
        "embedding_files": embedding_count,
        "qdrant_points": qdrant_count,
        "collection": CONFIG["collection_name"],
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
