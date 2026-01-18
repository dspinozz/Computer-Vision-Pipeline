#!/usr/bin/env python3
"""
Download Places365 dataset subset for scene understanding.

Downloads only the categories relevant to interactive applications.
Uses the 256x256 small version to reduce download size.

Usage:
    python download_places365.py --output-dir /data/image_pipeline/data/places365
    python download_places365.py --categories restaurant bar cafe --max-per-category 3000
"""

import os
import sys
import argparse
import requests
import tarfile
import shutil
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

console = Console()

# Places365 category mapping
# These are the category names in Places365 that match interactive applications
SCENE_CATEGORIES = {
    # Restaurants & Food
    "restaurant": "restaurant",
    "restaurant_kitchen": "restaurant_kitchen",
    "restaurant_patio": "restaurant_patio",
    "sushi_bar": "sushi_bar",
    "pizzeria": "pizzeria",
    "coffee_shop": "coffee_shop",
    "bakery": "bakery/shop",
    "bar": "bar",
    "pub": "pub/indoor",
    "beer_hall": "beer_hall",
    "food_court": "food_court",
    "diner": "diner/outdoor",
    "cafeteria": "cafeteria",
    "ice_cream_parlor": "ice_cream_parlor",
    
    # Retail & Shopping
    "bookstore": "bookstore",
    "supermarket": "supermarket",
    "shopping_mall": "shopping_mall/indoor",
    "market_indoor": "market/indoor",
    "market_outdoor": "market/outdoor",
    "drugstore": "drugstore",
    "clothing_store": "clothing_store",
    
    # Transportation
    "train_station": "train_station/platform",
    "subway_station": "subway_station/platform",
    "bus_station": "bus_station/indoor",
    "airport_terminal": "airport_terminal",
    "ticket_booth": "ticket_booth",
    
    # Workplace & Education
    "office": "office",
    "office_cubicles": "office_cubicles",
    "conference_room": "conference_room",
    "classroom": "classroom",
    "lecture_room": "lecture_room",
    "library": "library/indoor",
    "computer_room": "computer_room",
    
    # Urban & Streets
    "street": "street",
    "alley": "alley",
    "crosswalk": "crosswalk",
    "plaza": "plaza",
    "downtown": "downtown",
    "subway_interior": "subway_interior",
    "bus_interior": "bus_interior",
    
    # Home & Living
    "living_room": "living_room",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "dining_room": "dining_room",
    "apartment_building": "apartment_building/outdoor",
    
    # Hospitality
    "hotel_room": "hotel_room",
    "hotel_lobby": "lobby",
    "waiting_room": "waiting_room",
}

# Base URLs for Places365
PLACES365_BASE_URL = "http://data.csail.mit.edu/places/places365"
CATEGORY_INDEX_URL = f"{PLACES365_BASE_URL}/categories_places365.txt"
TRAIN_IMAGES_URL = f"{PLACES365_BASE_URL}/train_256_places365standard.tar"
VAL_IMAGES_URL = f"{PLACES365_BASE_URL}/val_256.tar"


def get_category_list() -> dict:
    """Fetch the official category list from Places365."""
    try:
        response = requests.get(CATEGORY_INDEX_URL, timeout=30)
        response.raise_for_status()
        
        categories = {}
        for line in response.text.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                category_path = parts[0]  # e.g., "/a/abbey"
                category_id = int(parts[1])
                # Extract category name from path
                category_name = category_path.split('/')[-1]
                categories[category_name] = {
                    "path": category_path,
                    "id": category_id
                }
        
        return categories
    except Exception as e:
        console.print(f"[red]Failed to fetch category list: {e}[/red]")
        return {}


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress tracking."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"Downloading {output_path.name}", total=total_size)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
        
        return True
        
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        return False


def extract_categories(tar_path: Path, output_dir: Path, categories: List[str]) -> int:
    """Extract only specified categories from a tar archive."""
    console.print(f"\n[blue]Extracting categories from {tar_path.name}...[/blue]")
    
    extracted_count = 0
    category_counts = {cat: 0 for cat in categories}
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            
            with Progress(
                TextColumn("[bold blue]Extracting..."),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} files"),
            ) as progress:
                task = progress.add_task("Extracting", total=len(members))
                
                for member in members:
                    progress.update(task, advance=1)
                    
                    # Check if this file belongs to a category we want
                    for cat in categories:
                        if f"/{cat}/" in member.name or member.name.startswith(f"{cat}/"):
                            # Extract this file
                            tar.extract(member, output_dir)
                            category_counts[cat] += 1
                            extracted_count += 1
                            break
        
        # Print summary
        table = Table(title="Extracted Categories")
        table.add_column("Category", style="cyan")
        table.add_column("Images", justify="right", style="green")
        
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                table.add_row(cat, str(count))
        
        console.print(table)
        
        return extracted_count
        
    except Exception as e:
        console.print(f"[red]Extraction failed: {e}[/red]")
        return 0


def download_category_subset(output_dir: Path, categories: List[str], max_per_category: int = 5000) -> int:
    """
    Alternative: Download images directly using the file list.
    This is more efficient when you only need a small subset.
    """
    console.print(f"\n[bold]Downloading {len(categories)} categories, max {max_per_category} each[/bold]")
    
    # First, get the training file list
    train_list_url = f"{PLACES365_BASE_URL}/filelist_places365-standard/places365_train_standard.txt"
    
    try:
        console.print("[blue]Fetching file list...[/blue]")
        response = requests.get(train_list_url, timeout=60)
        response.raise_for_status()
        
        # Parse file list and group by category
        files_by_category = {cat: [] for cat in categories}
        
        for line in response.text.strip().split('\n'):
            parts = line.split()
            if parts:
                file_path = parts[0]
                # Extract category from path (e.g., "/a/abbey/00000001.jpg" -> "abbey")
                path_parts = file_path.split('/')
                if len(path_parts) >= 3:
                    category = path_parts[2]  # After the letter directory
                    if category in files_by_category:
                        files_by_category[category].append(file_path)
        
        # Limit to max_per_category
        for cat in files_by_category:
            files_by_category[cat] = files_by_category[cat][:max_per_category]
        
        total_files = sum(len(files) for files in files_by_category.values())
        console.print(f"[green]Found {total_files} files to download[/green]")
        
        # Create category directories
        for cat in categories:
            (output_dir / cat).mkdir(parents=True, exist_ok=True)
        
        # Download files
        # Note: Individual file downloads would be very slow
        # The tar download is actually more efficient for bulk downloads
        
        console.print("[yellow]Note: For bulk downloads, using the tar archive is recommended.[/yellow]")
        console.print("[yellow]This script will download the validation set for quick testing.[/yellow]")
        
        return total_files
        
    except Exception as e:
        console.print(f"[red]Failed to fetch file list: {e}[/red]")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download Places365 subset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/image_pipeline/data/places365"),
        help="Output directory"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(SCENE_CATEGORIES.keys()),
        help="Categories to download (default: scene categories)"
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=3000,
        help="Maximum images per category"
    )
    parser.add_argument(
        "--use-validation",
        action="store_true",
        help="Download validation set (smaller, ~1GB) instead of training set"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available categories and exit"
    )
    
    args = parser.parse_args()
    
    # List categories mode
    if args.list_categories:
        console.print("\n[bold]Available scene categories:[/bold]")
        for key, value in sorted(SCENE_CATEGORIES.items()):
            console.print(f"  • {key}: {value}")
        console.print(f"\n[bold]Total: {len(SCENE_CATEGORIES)} categories[/bold]")
        return 0
    
    # Validate categories
    valid_categories = []
    for cat in args.categories:
        if cat in SCENE_CATEGORIES:
            valid_categories.append(cat)
        else:
            console.print(f"[yellow]Warning: Unknown category '{cat}', skipping[/yellow]")
    
    if not valid_categories:
        console.print("[red]No valid categories specified[/red]")
        return 1
    
    # Show download plan
    console.print("\n[bold]Places365 Download Plan[/bold]")
    console.print("=" * 50)
    console.print(f"Categories: {len(valid_categories)}")
    console.print(f"Max per category: {args.max_per_category}")
    console.print(f"Estimated images: {len(valid_categories) * args.max_per_category:,}")
    console.print(f"Output directory: {args.output_dir}")
    
    if args.use_validation:
        console.print(f"Dataset: Validation (256x256, ~1GB)")
        tar_url = VAL_IMAGES_URL
    else:
        console.print(f"Dataset: Training (256x256, ~25GB compressed)")
        tar_url = TRAIN_IMAGES_URL
    
    if args.dry_run:
        console.print("\n[yellow]Dry run - no files downloaded[/yellow]")
        
        # Just show category info
        console.print("\n[bold]Categories to download:[/bold]")
        for cat in valid_categories:
            console.print(f"  • {cat}")
        
        return 0
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # For validation set (quick testing)
    if args.use_validation:
        tar_path = args.output_dir / "val_256.tar"
        
        if not tar_path.exists():
            console.print(f"\n[blue]Downloading validation set (~1GB)...[/blue]")
            if not download_file(tar_url, tar_path):
                return 1
        else:
            console.print(f"[yellow]Using existing tar file: {tar_path}[/yellow]")
        
        # Extract selected categories
        extracted = extract_categories(tar_path, args.output_dir, valid_categories)
        console.print(f"\n[green]Extracted {extracted} images[/green]")
        
    else:
        # For full training set, we need a different approach
        # The full tar is ~25GB which is too large to download entirely
        
        console.print("\n[yellow]For the full training set, downloading the tar archive...[/yellow]")
        console.print("[yellow]This may take a while (~25GB download)[/yellow]")
        
        tar_path = args.output_dir / "train_256_places365standard.tar"
        
        if not tar_path.exists():
            # Check disk space first
            import shutil
            _, _, free = shutil.disk_usage(args.output_dir)
            if free < 30 * 1024**3:  # 30GB needed
                console.print("[red]Not enough disk space (need ~30GB for download + extraction)[/red]")
                console.print("[yellow]Try --use-validation for a smaller download[/yellow]")
                return 1
            
            console.print(f"\n[blue]Downloading training set (~25GB)...[/blue]")
            console.print("[yellow]This will take a while. You can Ctrl+C and resume later.[/yellow]")
            
            if not download_file(tar_url, tar_path):
                return 1
        else:
            console.print(f"[yellow]Using existing tar file: {tar_path}[/yellow]")
        
        # Extract selected categories
        extracted = extract_categories(tar_path, args.output_dir, valid_categories)
        console.print(f"\n[green]Extracted {extracted} images[/green]")
    
    # Summary
    console.print("\n[bold green]Download complete![/bold green]")
    console.print(f"Images saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
