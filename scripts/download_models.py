#!/usr/bin/env python3
"""
Download all required models for the image pipeline.

Models:
- Qwen2.5-VL-3B-Instruct (captioning, OCR)
- SigLIP-So400m (embeddings, tag verification)
- GroundingDINO-base (object localization)
- SAM2-tiny (mask generation)

Usage:
    python download_models.py --output-dir /data/image_pipeline/models
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Model specifications with expected sizes
MODELS = {
    "qwen2.5-vl-3b": {
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "description": "Qwen2.5-VL-3B for captioning and OCR",
        "expected_size_gb": 6.5,
        "subfolder": None,
    },
    "siglip-so400m": {
        "repo_id": "google/siglip-so400m-patch14-384",
        "description": "SigLIP for embeddings and tag verification",
        "expected_size_gb": 1.5,
        "subfolder": None,
    },
    "grounding-dino-base": {
        "repo_id": "IDEA-Research/grounding-dino-base",
        "description": "GroundingDINO for object localization",
        "expected_size_gb": 1.0,
        "subfolder": None,
    },
    "sam2-tiny": {
        "repo_id": "facebook/sam2-hiera-tiny",
        "description": "SAM2 for mask generation",
        "expected_size_gb": 0.15,
        "subfolder": None,
    },
}


def check_disk_space(path: Path, required_gb: float) -> bool:
    """Check if there's enough disk space."""
    import shutil
    
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    
    if free_gb < required_gb:
        console.print(f"[red]Error: Not enough disk space. Need {required_gb:.1f}GB, have {free_gb:.1f}GB[/red]")
        return False
    
    console.print(f"[green]Disk space OK: {free_gb:.1f}GB free, need {required_gb:.1f}GB[/green]")
    return True


def download_model(model_key: str, output_dir: Path, force: bool = False) -> bool:
    """Download a single model."""
    model_info = MODELS[model_key]
    model_dir = output_dir / model_key
    
    # Check if already downloaded
    if model_dir.exists() and not force:
        # Check if it looks complete (has config files)
        config_files = list(model_dir.glob("*.json")) + list(model_dir.glob("*.safetensors"))
        if config_files:
            console.print(f"[yellow]Skipping {model_key}: already exists at {model_dir}[/yellow]")
            return True
    
    console.print(f"\n[bold blue]Downloading {model_key}[/bold blue]")
    console.print(f"  Repository: {model_info['repo_id']}")
    console.print(f"  Expected size: ~{model_info['expected_size_gb']:.1f} GB")
    console.print(f"  Destination: {model_dir}")
    
    try:
        snapshot_download(
            repo_id=model_info["repo_id"],
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        console.print(f"[green]✓ Successfully downloaded {model_key}[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Failed to download {model_key}: {e}[/red]")
        return False


def get_total_expected_size() -> float:
    """Get total expected download size in GB."""
    return sum(m["expected_size_gb"] for m in MODELS.values())


def main():
    parser = argparse.ArgumentParser(description="Download models for image pipeline")
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("/data/image_pipeline/models"),
        help="Output directory for models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models exist"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading"
    )
    
    args = parser.parse_args()
    
    # Determine which models to download
    if "all" in args.models:
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = args.models
    
    # Calculate expected size
    total_size = sum(MODELS[m]["expected_size_gb"] for m in models_to_download)
    
    console.print("\n[bold]Model Download Plan[/bold]")
    console.print("=" * 50)
    
    for model_key in models_to_download:
        model_info = MODELS[model_key]
        console.print(f"  • {model_key}: {model_info['description']} (~{model_info['expected_size_gb']:.1f} GB)")
    
    console.print(f"\n[bold]Total expected download: ~{total_size:.1f} GB[/bold]")
    
    if args.dry_run:
        console.print("\n[yellow]Dry run - no files downloaded[/yellow]")
        return 0
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check disk space (with 20% buffer)
    if not check_disk_space(args.output_dir, total_size * 1.2):
        return 1
    
    # Download each model
    success_count = 0
    for model_key in models_to_download:
        if download_model(model_key, args.output_dir, args.force):
            success_count += 1
    
    console.print(f"\n[bold]Download complete: {success_count}/{len(models_to_download)} models[/bold]")
    
    if success_count == len(models_to_download):
        console.print("[green]All models downloaded successfully![/green]")
        return 0
    else:
        console.print("[red]Some models failed to download[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
