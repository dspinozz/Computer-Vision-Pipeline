#!/usr/bin/env python3
"""Migrate JSON metadata files to PostgreSQL.

Usage:
    python scripts/migrate_to_postgres.py
    python scripts/migrate_to_postgres.py --limit 100  # Test with subset
    python scripts/migrate_to_postgres.py --dry-run    # Validate only
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import uuid
import os

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("Install psycopg2: pip install psycopg2-binary")
    exit(1)

from tqdm import tqdm
from rich.console import Console

console = Console()

# Configuration
CONFIG = {
    "metadata_dir": Path("/data/image_pipeline/outputs/metadata"),
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "cvpipeline_images",
    "db_user": "cvpipeline",
    "db_password": os.environ.get("POSTGRES_PASSWORD", "cvpipeline_dev_password"),
}


def get_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=CONFIG["db_host"],
        port=CONFIG["db_port"],
        dbname=CONFIG["db_name"],
        user=CONFIG["db_user"],
        password=CONFIG["db_password"],
    )


def load_metadata_files(metadata_dir: Path, limit: int = None) -> List[Dict]:
    """Load all JSON metadata files."""
    files = list(metadata_dir.glob("*.json"))
    if limit:
        files = files[:limit]
    
    metadata_list = []
    for f in tqdm(files, desc="Loading JSON files"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                metadata_list.append(data)
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[yellow]Warning: Could not load {f.name}: {e}[/yellow]")
    
    return metadata_list


def migrate_images(conn, metadata_list: List[Dict], dry_run: bool = False) -> int:
    """Migrate images table."""
    console.print("[blue]Migrating images...[/blue]")
    
    rows = []
    for m in metadata_list:
        rows.append((
            m["image_id"],
            m["file_hash"],
            m["file_path"],
            m["width"],
            m["height"],
            m.get("source_type", "unknown"),
            m.get("source_category"),
            m.get("embedding", {}).get("point_id") if m.get("embedding") else None,
            m.get("processed_at"),
        ))
    
    if dry_run:
        console.print(f"  Would insert {len(rows)} images")
        return len(rows)
    
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO images (id, file_hash, file_path, width, height, 
                               source_type, source_category, embedding_id, processed_at)
            VALUES %s
            ON CONFLICT (file_hash) DO UPDATE SET
                file_path = EXCLUDED.file_path,
                width = EXCLUDED.width,
                height = EXCLUDED.height,
                source_type = EXCLUDED.source_type,
                source_category = EXCLUDED.source_category,
                embedding_id = EXCLUDED.embedding_id,
                processed_at = EXCLUDED.processed_at,
                updated_at = NOW()
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            page_size=500,
        )
    conn.commit()
    
    return len(rows)


def migrate_captions(conn, metadata_list: List[Dict], dry_run: bool = False) -> int:
    """Migrate captions table."""
    console.print("[blue]Migrating captions...[/blue]")
    
    rows = []
    for m in metadata_list:
        caption = m.get("caption", {})
        if not caption:
            continue
        
        rows.append((
            m["image_id"],
            caption.get("text"),
            caption.get("short"),
            caption.get("extracted_nouns", []),
            caption.get("ocr_text"),
            caption.get("language_hint"),
            caption.get("model", "qwen2.5-vl-3b"),
        ))
    
    if dry_run:
        console.print(f"  Would insert {len(rows)} captions")
        return len(rows)
    
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO captions (image_id, caption_full, caption_short, 
                                 extracted_nouns, ocr_text, language_hint, model_name)
            VALUES %s
            ON CONFLICT (image_id) DO UPDATE SET
                caption_full = EXCLUDED.caption_full,
                caption_short = EXCLUDED.caption_short,
                extracted_nouns = EXCLUDED.extracted_nouns,
                ocr_text = EXCLUDED.ocr_text,
                language_hint = EXCLUDED.language_hint,
                model_name = EXCLUDED.model_name
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s)",
            page_size=500,
        )
    conn.commit()
    
    return len(rows)


def migrate_tags(conn, metadata_list: List[Dict], dry_run: bool = False) -> int:
    """Migrate tags table."""
    console.print("[blue]Migrating tags...[/blue]")
    
    rows = []
    for m in metadata_list:
        tags = m.get("tags", [])
        for i, tag in enumerate(tags):
            rows.append((
                m["image_id"],
                tag.get("tag_id"),
                tag.get("display", tag.get("tag_id")),
                tag.get("category", "unknown"),
                tag.get("confidence", 0),
                i + 1,  # rank
                tag.get("source", "siglip_cosine"),
                json.dumps(tag.get("evidence", {})),
            ))
    
    if dry_run:
        console.print(f"  Would insert {len(rows)} tags")
        return len(rows)
    
    # Clear existing tags and reinsert (simpler than upsert with rank)
    image_ids = list(set(m["image_id"] for m in metadata_list))
    with conn.cursor() as cur:
        # Delete in batches to avoid param limits
        for i in range(0, len(image_ids), 100):
            batch = image_ids[i:i+100]
            cur.execute(
                "DELETE FROM tags WHERE image_id = ANY(%s::uuid[])",
                (batch,)
            )
        
        execute_values(
            cur,
            """
            INSERT INTO tags (image_id, tag_id, display_name, category, 
                             confidence, rank, source, evidence)
            VALUES %s
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s, %s)",
            page_size=1000,
        )
    conn.commit()
    
    return len(rows)


def migrate_detections(conn, metadata_list: List[Dict], dry_run: bool = False) -> int:
    """Migrate detections table (Lane B objects)."""
    console.print("[blue]Migrating detections...[/blue]")
    
    rows = []
    for m in metadata_list:
        objects = m.get("objects") or []
        for obj in objects:
            box = obj.get("box_xyxy", [0, 0, 0, 0])
            rows.append((
                m["image_id"],
                obj.get("label"),
                obj.get("confidence", 0),
                box[0], box[1], box[2], box[3],
                "xyxy_pixels",
                obj.get("area_px"),
            ))
    
    if dry_run:
        console.print(f"  Would insert {len(rows)} detections")
        return len(rows)
    
    if not rows:
        console.print("  No detections to migrate")
        return 0
    
    # Clear and reinsert
    image_ids = list(set(m["image_id"] for m in metadata_list if m.get("objects")))
    with conn.cursor() as cur:
        if image_ids:
            for i in range(0, len(image_ids), 100):
                batch = image_ids[i:i+100]
                cur.execute(
                    "DELETE FROM detections WHERE image_id = ANY(%s::uuid[])",
                    (batch,)
                )
        
        if rows:
            execute_values(
                cur,
                """
                INSERT INTO detections (image_id, label, confidence, 
                                       box_x1, box_y1, box_x2, box_y2, 
                                       box_format, area_px)
                VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                page_size=1000,
            )
    conn.commit()
    
    return len(rows)


def migrate_masks(conn, metadata_list: List[Dict], dry_run: bool = False) -> int:
    """Migrate masks table (Lane B masks)."""
    console.print("[blue]Migrating masks...[/blue]")
    
    rows = []
    for m in metadata_list:
        objects = m.get("objects") or []
        for obj in objects:
            mask = obj.get("mask")
            if not mask:
                continue
            
            rows.append((
                m["image_id"],
                None,  # detection_id - would need to look up
                obj.get("label"),
                mask.get("rle", ""),
                "coco_rle",
                mask.get("area_px"),
                mask.get("iou_score"),
            ))
    
    if dry_run:
        console.print(f"  Would insert {len(rows)} masks")
        return len(rows)
    
    if not rows:
        console.print("  No masks to migrate")
        return 0
    
    # Clear and reinsert
    image_ids = list(set(
        m["image_id"] for m in metadata_list 
        if any(obj.get("mask") for obj in (m.get("objects") or []))
    ))
    with conn.cursor() as cur:
        if image_ids:
            for i in range(0, len(image_ids), 100):
                batch = image_ids[i:i+100]
                cur.execute(
                    "DELETE FROM masks WHERE image_id = ANY(%s::uuid[])",
                    (batch,)
                )
        
        if rows:
            execute_values(
                cur,
                """
                INSERT INTO masks (image_id, detection_id, label, 
                                  mask_rle, mask_format, area_px, iou_score)
                VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s)",
                page_size=1000,
            )
    conn.commit()
    
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON metadata to Postgres")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't insert")
    args = parser.parse_args()
    
    console.print("\n[bold blue]Migrating Image Metadata to PostgreSQL[/bold blue]")
    console.print(f"Source: {CONFIG['metadata_dir']}")
    console.print(f"Database: {CONFIG['db_name']}@{CONFIG['db_host']}:{CONFIG['db_port']}")
    
    # Load metadata
    console.print("\n[yellow]Loading metadata files...[/yellow]")
    metadata_list = load_metadata_files(CONFIG["metadata_dir"], args.limit)
    console.print(f"Loaded {len(metadata_list)} metadata records")
    
    if not metadata_list:
        console.print("[red]No metadata files found![/red]")
        return
    
    if args.dry_run:
        console.print("\n[yellow]DRY RUN - No data will be written[/yellow]")
    
    # Connect to database
    try:
        conn = get_connection()
        console.print("[green]✓ Connected to database[/green]")
    except Exception as e:
        console.print(f"[red]Failed to connect to database: {e}[/red]")
        console.print("\nMake sure Postgres is running:")
        console.print("  cd /data/image_pipeline && docker compose up -d postgres")
        return
    
    # Run migrations
    try:
        images_count = migrate_images(conn, metadata_list, args.dry_run)
        captions_count = migrate_captions(conn, metadata_list, args.dry_run)
        tags_count = migrate_tags(conn, metadata_list, args.dry_run)
        detections_count = migrate_detections(conn, metadata_list, args.dry_run)
        masks_count = migrate_masks(conn, metadata_list, args.dry_run)
        
        console.print("\n[bold green]Migration Complete![/bold green]")
        console.print(f"  Images: {images_count}")
        console.print(f"  Captions: {captions_count}")
        console.print(f"  Tags: {tags_count}")
        console.print(f"  Detections: {detections_count}")
        console.print(f"  Masks: {masks_count}")
        
        # Verify counts
        if not args.dry_run:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM images")
                total_images = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM tags")
                total_tags = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM detections")
                total_detections = cur.fetchone()[0]
                
            console.print("\n[bold]Database Statistics:[/bold]")
            console.print(f"  Total images in DB: {total_images}")
            console.print(f"  Total tags in DB: {total_tags}")
            console.print(f"  Total detections in DB: {total_detections}")
        
    except Exception as e:
        console.print(f"[red]Migration error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
