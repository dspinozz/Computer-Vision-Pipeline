#!/usr/bin/env python3
"""
Qdrant Import Script

Imports embeddings from index.jsonl + .npy files into Qdrant.
Creates collection if it doesn't exist.

Usage:
    # Start Qdrant first:
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    
    # Then import:
    python scripts/import_qdrant.py
    python scripts/import_qdrant.py --recreate  # Drop and recreate collection
    python scripts/import_qdrant.py --dry-run   # Just validate, don't import
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, Distance, PointStruct,
        OptimizersConfigDiff, HnswConfigDiff
    )
except ImportError:
    print("Install qdrant-client: pip install qdrant-client")
    exit(1)

# Configuration
CONFIG = {
    "qdrant_url": "http://localhost:6333",
    "collection_name": "image_embeddings",
    "embedding_dim": 1152,  # SigLIP So400m
    "batch_size": 100,
    "index_path": Path("/data/image_pipeline/outputs/embeddings/index.jsonl"),
    "metadata_dir": Path("/data/image_pipeline/outputs/metadata"),
}


def load_index(index_path: Path) -> List[Dict]:
    """Load embedding index from JSONL file."""
    entries = []
    with open(index_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_metadata(metadata_dir: Path, file_hash: str) -> Optional[Dict]:
    """Load metadata for an image by searching for matching file_hash."""
    # We need to find the metadata file that contains this hash
    # This is a bit inefficient but works for now
    # In production, we'd have a hash->file index
    for meta_file in metadata_dir.glob("*.json"):
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            if meta.get("file_hash") == file_hash:
                return meta
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def build_hash_to_metadata_index(metadata_dir: Path) -> Dict[str, Dict]:
    """Build index of file_hash -> metadata for fast lookup."""
    print("Building metadata index...")
    index = {}
    for meta_file in tqdm(list(metadata_dir.glob("*.json")), desc="Indexing metadata"):
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            index[meta.get("file_hash")] = meta
        except (json.JSONDecodeError, KeyError):
            continue
    return index


def create_collection(client: QdrantClient, collection_name: str, dim: int, recreate: bool = False):
    """Create Qdrant collection with optimal settings for image search."""
    
    collections = [c.name for c in client.get_collections().collections]
    
    if collection_name in collections:
        if recreate:
            print(f"Dropping existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists. Use --recreate to drop and recreate.")
            return False
    
    print(f"Creating collection: {collection_name} (dim={dim})")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,  # SigLIP embeddings work best with cosine
        ),
        # Optimize for search speed
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,  # Start indexing after 20k points
        ),
    )
    
    return True


def import_embeddings(
    client: QdrantClient,
    collection_name: str,
    index_entries: List[Dict],
    metadata_index: Dict[str, Dict],
    batch_size: int = 100,
    dry_run: bool = False,
):
    """Import embeddings into Qdrant with metadata payload."""
    
    points_batch = []
    skipped = 0
    imported = 0
    
    for entry in tqdm(index_entries, desc="Importing"):
        file_hash = entry["file_hash"]
        point_id = entry["point_id"]
        embedding_file = Path(entry["embedding_file"])
        
        # Load embedding
        if not embedding_file.exists():
            skipped += 1
            continue
        
        embedding = np.load(embedding_file)
        if embedding.ndim > 1:
            embedding = embedding[0]
        
        # Get metadata for payload
        meta = metadata_index.get(file_hash, {})
        
        # Build payload (searchable/filterable fields)
        payload = {
            "image_id": meta.get("image_id", point_id),
            "file_hash": file_hash,
            "file_path": meta.get("file_path", entry.get("image_path", "")),
            "source_type": meta.get("source_type", "unknown"),
            "source_category": meta.get("source_category", "unknown"),
            "width": meta.get("width", 0),
            "height": meta.get("height", 0),
            # Caption for text search
            "caption_short": meta.get("caption", {}).get("short", ""),
            "caption_full": meta.get("caption", {}).get("text", ""),
            # Top tags for filtering
            "tags": [t.get("tag_id") for t in (meta.get("tags") or [])[:5]],
            "tag_displays": [t.get("display") for t in (meta.get("tags") or [])[:5]],
            # Top tag scores for ranking
            "top_tag_score": (meta.get("tags") or [{}])[0].get("confidence", 0) if meta.get("tags") else 0,
        }
        
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload,
        )
        
        points_batch.append(point)
        
        # Upsert in batches
        if len(points_batch) >= batch_size:
            if not dry_run:
                client.upsert(collection_name=collection_name, points=points_batch)
            imported += len(points_batch)
            points_batch = []
    
    # Final batch
    if points_batch:
        if not dry_run:
            client.upsert(collection_name=collection_name, points=points_batch)
        imported += len(points_batch)
    
    return imported, skipped


def main():
    parser = argparse.ArgumentParser(description="Import embeddings to Qdrant")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't import")
    parser.add_argument("--url", default=CONFIG["qdrant_url"], help="Qdrant URL")
    parser.add_argument("--collection", default=CONFIG["collection_name"], help="Collection name")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="Batch size")
    args = parser.parse_args()
    
    # Load index
    print(f"Loading index from {CONFIG['index_path']}")
    index_entries = load_index(CONFIG["index_path"])
    print(f"Found {len(index_entries)} embeddings to import")
    
    if not index_entries:
        print("No embeddings to import!")
        return
    
    # Build metadata index for fast lookup
    metadata_index = build_hash_to_metadata_index(CONFIG["metadata_dir"])
    print(f"Loaded {len(metadata_index)} metadata entries")
    
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would import {len(index_entries)} embeddings")
        print(f"Metadata coverage: {len(metadata_index)}/{len(index_entries)} ({len(metadata_index)/len(index_entries)*100:.1f}%)")
        
        # Check a sample
        sample = index_entries[0]
        emb_path = Path(sample["embedding_file"])
        print(f"\nSample entry:")
        print(f"  point_id: {sample['point_id']}")
        print(f"  file_hash: {sample['file_hash'][:20]}...")
        print(f"  embedding exists: {emb_path.exists()}")
        if emb_path.exists():
            emb = np.load(emb_path)
            print(f"  embedding shape: {emb.shape}")
        return
    
    # Connect to Qdrant
    print(f"\nConnecting to Qdrant at {args.url}")
    try:
        client = QdrantClient(url=args.url)
        # Test connection
        client.get_collections()
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        return
    
    # Create collection
    create_collection(
        client, 
        args.collection, 
        CONFIG["embedding_dim"], 
        recreate=args.recreate
    )
    
    # Import embeddings
    imported, skipped = import_embeddings(
        client,
        args.collection,
        index_entries,
        metadata_index,
        batch_size=args.batch_size,
    )
    
    print(f"\n=== Import Complete ===")
    print(f"Imported: {imported}")
    print(f"Skipped (missing embedding): {skipped}")
    
    # Verify
    info = client.get_collection(args.collection)
    print(f"Collection '{args.collection}' now has {info.points_count} points")


if __name__ == "__main__":
    main()
