#!/usr/bin/env python3
"""
Simple test script to verify each model loads and runs.
Run this before the full pipeline to catch issues early.

Usage:
    python test_models.py --model siglip
    python test_models.py --model qwen
    python test_models.py --model dino
    python test_models.py --all
"""

import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

MODELS_DIR = Path("/data/image_pipeline/models")
TEST_IMAGE = Path("/data/image_pipeline/data/places365/val_256/Places365_val_00000001.jpg")


def test_siglip():
    """Test SigLIP embedding model."""
    console.print("\n[bold blue]Testing SigLIP...[/bold blue]")
    
    try:
        import torch
        from transformers import AutoProcessor, AutoModel
        from PIL import Image
        
        model_path = MODELS_DIR / "siglip-so400m"
        
        console.print(f"  Loading from {model_path}")
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        console.print("  Running inference...")
        image = Image.open(TEST_IMAGE).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            embedding = outputs[0].cpu().numpy()
        
        console.print(f"  [green]✓ Success! Embedding shape: {embedding.shape}[/green]")
        return True
        
    except Exception as e:
        console.print(f"  [red]✗ Failed: {e}[/red]")
        return False


def test_qwen():
    """Test Qwen2.5-VL captioning model."""
    console.print("\n[bold blue]Testing Qwen2.5-VL...[/bold blue]")
    
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from PIL import Image
        
        model_path = MODELS_DIR / "qwen2.5-vl-3b"
        
        console.print(f"  Loading from {model_path}")
        processor = AutoProcessor.from_pretrained(model_path)
        # Load in fp16 without int8 for testing - int8 can be added later
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        
        console.print("  Running inference...")
        image = Image.open(TEST_IMAGE).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in one sentence."}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        output_text = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        console.print(f"  [green]✓ Success! Caption: {output_text[:100]}...[/green]")
        return True
        
    except Exception as e:
        console.print(f"  [red]✗ Failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def test_grounding_dino():
    """Test GroundingDINO object detection model."""
    console.print("\n[bold blue]Testing GroundingDINO...[/bold blue]")
    
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from PIL import Image
        
        model_path = MODELS_DIR / "grounding-dino-base"
        
        console.print(f"  Loading from {model_path}")
        processor = AutoProcessor.from_pretrained(model_path)
        # Use float32 for stability - GroundingDINO has dtype issues with fp16
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        model.eval()
        
        console.print("  Running inference...")
        image = Image.open(TEST_IMAGE).convert("RGB")
        text = "person. table. chair. window."
        
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process - API changed in newer transformers
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            target_sizes=[(image.height, image.width)]
        )[0]
        
        # Filter by threshold manually
        threshold = 0.3
        mask = results["scores"] >= threshold
        scores = results["scores"][mask]
        labels = [results["labels"][i] for i, m in enumerate(mask) if m]
        boxes = results["boxes"][mask]
        
        num_objects = len(scores)
        console.print(f"  [green]✓ Success! Detected {num_objects} objects (threshold={threshold})[/green]")
        
        for score, label, box in zip(scores[:5], labels[:5], boxes[:5]):
            console.print(f"    - {label}: {score:.2f}")
        
        return True
        
    except Exception as e:
        console.print(f"  [red]✗ Failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def test_sam2():
    """Test SAM2 mask generation model."""
    console.print("\n[bold blue]Testing SAM2...[/bold blue]")
    
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
        from PIL import Image
        
        model_path = MODELS_DIR / "sam2-tiny"
        
        console.print(f"  Loading from {model_path}")
        processor = Sam2Processor.from_pretrained(model_path)
        # Use float32 for stability (SAM2 has dtype issues with fp16)
        model = Sam2Model.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        model.eval()
        
        console.print("  Running inference...")
        image = Image.open(TEST_IMAGE).convert("RGB")
        
        # SAM2 requires 4 levels: [image, object, point, coords]
        input_points = [[[[image.width // 2, image.height // 2]]]]
        
        inputs = processor(images=image, input_points=input_points, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # outputs.pred_masks shape: [batch, objects, num_masks, H, W]
        mask_shape = outputs.pred_masks.shape
        iou_scores = outputs.iou_scores[0, 0].cpu().numpy()
        best_mask_idx = iou_scores.argmax()
        
        console.print(f"  [green]✓ Success! Mask shape: {mask_shape}, best IOU: {iou_scores[best_mask_idx]:.3f}[/green]")
        return True
        
    except Exception as e:
        console.print(f"  [red]✗ Failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument("--model", choices=["siglip", "qwen", "dino", "sam2"], help="Model to test")
    parser.add_argument("--all", action="store_true", help="Test all models")
    
    args = parser.parse_args()
    
    if not TEST_IMAGE.exists():
        console.print(f"[red]Test image not found: {TEST_IMAGE}[/red]")
        console.print("Run the Places365 download first.")
        return 1
    
    results = {}
    
    if args.all or args.model == "siglip":
        results["siglip"] = test_siglip()
    
    if args.all or args.model == "qwen":
        results["qwen"] = test_qwen()
    
    if args.all or args.model == "dino":
        results["dino"] = test_grounding_dino()
    
    if args.all or args.model == "sam2":
        results["sam2"] = test_sam2()
    
    if not results:
        parser.print_help()
        return 1
    
    # Summary
    console.print("\n[bold]Summary[/bold]")
    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="green")
    
    for model, success in results.items():
        status = "[green]✓ Pass[/green]" if success else "[red]✗ Fail[/red]"
        table.add_row(model, status)
    
    console.print(table)
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
