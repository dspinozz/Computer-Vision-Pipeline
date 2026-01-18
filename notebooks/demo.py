#!/usr/bin/env python3
"""
Vision Metadata Pipeline - Demo Script

Demonstrates:
1. Semantic search via Qdrant
2. Object detection visualization
3. Mask visualization
4. Caption and tag quality

Usage:
    python notebooks/demo.py
"""

import json
import requests
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
METADATA_DIR = Path("/data/image_pipeline/outputs/metadata")


def demo_semantic_search():
    """Demonstrate semantic search."""
    print("=" * 70)
    print("1. SEMANTIC SEARCH DEMO")
    print("=" * 70)
    
    queries = [
        "mountain landscape with snow",
        "people eating at restaurant", 
        "bedroom with wooden furniture",
        "beach sunset",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            resp = requests.get(f"{API_URL}/v1/images/search", params={"q": query, "limit": 2})
            data = resp.json()
            for i, result in enumerate(data.get("results", []), 1):
                print(f"   {i}. Score: {result['score']:.3f}")
                print(f"      Caption: {result['caption_short'][:60]}...")
                print(f"      Tags: {', '.join(result['tags'][:3])}")
        except Exception as e:
            print(f"   Error: {e}")


def demo_object_detection():
    """Demonstrate object detection data."""
    print("\n" + "=" * 70)
    print("2. OBJECT DETECTION DEMO")
    print("=" * 70)
    
    # Find files with multiple objects
    for f in list(METADATA_DIR.glob("*.json"))[:500]:
        with open(f) as fp:
            data = json.load(fp)
        objs = data.get("objects", [])
        if len(objs) >= 3:
            print(f"\n🖼️ {data['file_path']}")
            print(f"   Caption: {data.get('caption', {}).get('short', '')[:60]}...")
            print(f"   Objects detected: {len(objs)}")
            for obj in objs[:5]:
                label = obj['label']
                conf = obj['confidence']
                box = obj.get('box', {})
                mask_area = obj.get('mask_area', 0)
                print(f"   📦 {label.upper()}")
                print(f"      Confidence: {conf:.1%}")
                print(f"      BBox: ({box.get('x1',0):.2f}, {box.get('y1',0):.2f}) → ({box.get('x2',0):.2f}, {box.get('y2',0):.2f})")
                print(f"      Mask pixels: {mask_area}")
            break


def demo_mask_data():
    """Demonstrate mask data structure."""
    print("\n" + "=" * 70)
    print("3. MASK DATA DEMO")
    print("=" * 70)
    
    for f in list(METADATA_DIR.glob("*.json"))[:500]:
        with open(f) as fp:
            data = json.load(fp)
        objs = data.get("objects", [])
        for obj in objs:
            if obj.get('mask_rle') and obj.get('mask_area', 0) > 1000:
                print(f"\n🎭 Object: {obj['label']}")
                print(f"   Image: {data['file_path']}")
                print(f"   Mask area: {obj['mask_area']} pixels")
                print(f"   Mask IoU: {obj.get('mask_iou', 0):.2f}")
                print(f"   Centroid: {obj.get('centroid', [])}")
                print(f"   RLE (first 50 chars): {obj['mask_rle'][:50]}...")
                
                # Explain RLE format
                print(f"\n   ℹ️ RLE Format: Run-Length Encoding")
                print(f"      Each pair (start, length) marks a mask run")
                print(f"      Can be decoded to binary mask for rendering")
                return


def demo_caption_quality():
    """Demonstrate caption quality."""
    print("\n" + "=" * 70)
    print("4. CAPTION QUALITY DEMO")
    print("=" * 70)
    
    print("\nSample captions (VLM-generated):")
    for i, f in enumerate(list(METADATA_DIR.glob("*.json"))[:5], 1):
        with open(f) as fp:
            data = json.load(fp)
        caption = data.get("caption", {})
        if isinstance(caption, dict):
            print(f"\n{i}. {data['file_path'].split('/')[-1]}")
            print(f"   Full: {caption.get('text', '')[:150]}...")
            print(f"   Nouns: {caption.get('extracted_nouns', [])[:5]}")


def demo_tag_ranking():
    """Demonstrate tag ranking quality."""
    print("\n" + "=" * 70)
    print("5. TAG RANKING DEMO")
    print("=" * 70)
    
    print("\nTop tags are relevant (relative ranking works):")
    for f in list(METADATA_DIR.glob("*.json"))[:3]:
        with open(f) as fp:
            data = json.load(fp)
        
        print(f"\n🖼️ {data['file_path'].split('/')[-1]}")
        caption = data.get('caption', {})
        if isinstance(caption, dict):
            print(f"   Scene: {caption.get('short', '')[:50]}...")
        
        tags = data.get("tags", [])
        if tags and isinstance(tags[0], dict):
            sorted_tags = sorted(tags, key=lambda x: x.get('confidence', 0), reverse=True)
            print(f"   Top 3 tags:")
            for t in sorted_tags[:3]:
                print(f"      - {t.get('display', '?')}: {t.get('confidence', 0):.1%}")


def demo_stats():
    """Show collection statistics."""
    print("\n" + "=" * 70)
    print("6. COLLECTION STATISTICS")
    print("=" * 70)
    
    try:
        resp = requests.get(f"{API_URL}/v1/stats")
        data = resp.json()
        print(f"\n   Metadata files: {data.get('metadata_files', 'N/A')}")
        print(f"   Embedding files: {data.get('embedding_files', 'N/A')}")
        print(f"   Qdrant points: {data.get('qdrant_points', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    print("\n" + "🚀 VISION METADATA PIPELINE DEMO 🚀".center(70))
    print("=" * 70)
    
    demo_semantic_search()
    demo_object_detection()
    demo_mask_data()
    demo_caption_quality()
    demo_tag_ranking()
    demo_stats()
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)
