#!/usr/bin/env python3
"""
Image Processing Pipeline

Lane A (Base Library - run on all images):
1. Embedding (SigLIP) → Qdrant
2. Caption (Qwen2.5-VL) → Extract structured data
3. Tag (CLIP scoring against taxonomy) → Verified tags

Lane B (Interactive - on-demand or batch for library):
4. Object Detection (GroundingDINO) → Bounding boxes
5. Mask Generation (SAM2) → Pixel masks

Usage:
    python pipeline.py ingest --input-dir /data/image_pipeline/data/places365
    python pipeline.py process --lane A  # Just embeddings + captions + tags
    python pipeline.py process --lane B  # Add object detection + masks
    python pipeline.py process --lane all  # Full pipeline
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

from PIL import Image
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

# Configuration
CONFIG = {
    "models_dir": Path("/data/image_pipeline/models"),
    "data_dir": Path("/data/image_pipeline/data"),
    "output_dir": Path("/data/image_pipeline/outputs"),
    "taxonomy_path": Path("/data/image_pipeline/configs/taxonomy.json"),
    "batch_size": 1,  # Process one at a time to manage VRAM
    "max_caption_length": 200,
    "tag_confidence_threshold": 0.5,
    "object_confidence_threshold": 0.3,
    "max_objects_per_image": 20,
}


@dataclass
class ImageMetadata:
    """Complete metadata for a processed image."""
    image_id: str
    file_hash: str
    file_path: str
    width: int
    height: int
    source_type: str
    source_category: str
    caption: Optional[Dict] = None
    tags: Optional[List[Dict]] = None
    objects: Optional[List[Dict]] = None
    embedding_id: Optional[str] = None
    processed_at: Optional[str] = None
    processing_version: str = "1.0.0"


class TaxonomyMapper:
    """Maps extracted phrases to canonical taxonomy tags."""
    
    def __init__(self, taxonomy_path: Path):
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)
        
        # Build lookup tables
        self.tag_prompts = {}  # tag_id -> list of prompts
        self.phrase_to_tag = {}  # lowercase phrase -> tag_id
        self.tag_info = {}  # tag_id -> {display, category, localizable}
        
        self._build_lookups()
    
    def _build_lookups(self):
        """Build lookup tables from taxonomy."""
        for category, category_data in self.taxonomy.items():
            if category in ["version", "description"]:
                continue
            
            localizable = category_data.get("localizable", False)
            tags = category_data.get("tags", {})
            
            self._process_tags(tags, category, "", localizable)
    
    def _process_tags(self, tags: Dict, category: str, prefix: str, localizable: bool):
        """Recursively process tags and their children."""
        for tag_key, tag_data in tags.items():
            tag_id = f"{category}.{prefix}{tag_key}" if prefix else f"{category}.{tag_key}"
            
            # Store tag info
            self.tag_info[tag_id] = {
                "display": tag_data.get("display", tag_key),
                "category": category,
                "localizable": localizable,
            }
            
            # Store prompts
            prompts = tag_data.get("prompts", [])
            self.tag_prompts[tag_id] = prompts
            
            # Map phrases to tag
            for prompt in prompts:
                # Extract key phrases
                self.phrase_to_tag[prompt.lower()] = tag_id
                # Also map the display name
                self.phrase_to_tag[tag_data.get("display", tag_key).lower()] = tag_id
            
            # Process children
            children = tag_data.get("children", {})
            if children:
                self._process_tags(children, category, f"{tag_key}.", localizable)
    
    def map_phrase(self, phrase: str) -> Optional[str]:
        """Map a phrase to a canonical tag ID."""
        phrase_lower = phrase.lower().strip()
        
        # Direct match
        if phrase_lower in self.phrase_to_tag:
            return self.phrase_to_tag[phrase_lower]
        
        # Partial match
        for known_phrase, tag_id in self.phrase_to_tag.items():
            if known_phrase in phrase_lower or phrase_lower in known_phrase:
                return tag_id
        
        return None
    
    def get_localizable_tags(self) -> List[str]:
        """Get all tags that can be localized (for GroundingDINO)."""
        return [
            tag_id for tag_id, info in self.tag_info.items()
            if info.get("localizable", False)
        ]
    
    def get_all_prompts(self) -> Dict[str, List[str]]:
        """Get all tag prompts for CLIP scoring."""
        return self.tag_prompts


class ModelManager:
    """Manages loading/unloading models to fit in VRAM."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.device = "cuda"
    
    def load_siglip(self):
        """Load SigLIP model for embeddings."""
        if "siglip" in self.loaded_models:
            return self.loaded_models["siglip"]
        
        console.print("[blue]Loading SigLIP model...[/blue]")
        
        from transformers import AutoProcessor, AutoModel
        import torch
        
        model_path = self.models_dir / "siglip-so400m"
        
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        self.loaded_models["siglip"] = {
            "model": model,
            "processor": processor,
        }
        
        console.print("[green]SigLIP loaded[/green]")
        return self.loaded_models["siglip"]
    
    def load_qwen(self):
        """Load Qwen2.5-VL for captioning."""
        if "qwen" in self.loaded_models:
            return self.loaded_models["qwen"]
        
        console.print("[blue]Loading Qwen2.5-VL model (FP16)...[/blue]")
        
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        
        model_path = self.models_dir / "qwen2.5-vl-3b"
        
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        
        self.loaded_models["qwen"] = {
            "model": model,
            "processor": processor,
        }
        
        console.print("[green]Qwen2.5-VL loaded (FP16)[/green]")
        return self.loaded_models["qwen"]
    
    def load_grounding_dino(self):
        """Load GroundingDINO for object detection."""
        if "dino" in self.loaded_models:
            return self.loaded_models["dino"]
        
        console.print("[blue]Loading GroundingDINO...[/blue]")
        
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        
        model_path = self.models_dir / "grounding-dino-base"
        
        processor = AutoProcessor.from_pretrained(model_path)
        # Use float32 for stability - GroundingDINO has dtype issues with fp16
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        model.eval()
        
        self.loaded_models["dino"] = {
            "model": model,
            "processor": processor,
        }
        
        console.print("[green]GroundingDINO loaded[/green]")
        return self.loaded_models["dino"]
    
    def load_sam2(self):
        """Load SAM2 for mask generation."""
        if "sam2" in self.loaded_models:
            return self.loaded_models["sam2"]
        
        console.print("[blue]Loading SAM2...[/blue]")
        
        try:
            from transformers import Sam2Model, Sam2Processor
            import torch
            
            model_path = self.models_dir / "sam2-tiny"
            
            processor = Sam2Processor.from_pretrained(model_path)
            # Use float32 for stability (SAM2 has dtype issues with fp16)
            model = Sam2Model.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            model.eval()
            
            self.loaded_models["sam2"] = {
                "model": model,
                "processor": processor,
            }
            
            console.print("[green]SAM2 loaded[/green]")
        except Exception as e:
            console.print(f"[yellow]SAM2 not available: {e}[/yellow]")
            self.loaded_models["sam2"] = None
        
        return self.loaded_models.get("sam2")
    
    def unload(self, model_name: str):
        """Unload a model to free VRAM."""
        if model_name in self.loaded_models:
            import torch
            del self.loaded_models[model_name]
            torch.cuda.empty_cache()
            console.print(f"[yellow]Unloaded {model_name}[/yellow]")


class Pipeline:
    """Main processing pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_manager = ModelManager(config["models_dir"])
        self.taxonomy = TaxonomyMapper(config["taxonomy_path"])
        # Cache for taxonomy prompt text embeddings (SigLIP). Built lazily on first `score_tags` call.
        self._taxonomy_prompt_cache = None
        # Stable mapping from file_hash -> point_id (so embeddings can be regenerated without changing IDs)
        self._embedding_id_by_hash = None
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def get_image_info(self, image_path: Path) -> Dict:
        """Get basic image information."""
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format.lower() if img.format else "unknown",
            }
    
    def generate_embedding(self, image_path: Path, file_hash: str) -> Tuple[List[float], str]:
        """Generate image embedding using SigLIP and save to file.
        
        Idempotent:
        - Uses `file_hash` as a stable point id.
        - If the `.npy` already exists, loads it instead of recomputing.
        - Only appends to `index.jsonl` when creating a new embedding file.
        """
        import torch
        import numpy as np
        
        siglip = self.model_manager.load_siglip()
        model = siglip["model"]
        processor = siglip["processor"]

        # Stable point ID (so re-runs don't create duplicates)
        embeddings_dir = self.config["output_dir"] / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        point_id_map_path = embeddings_dir / "point_ids.json"
        
        if self._embedding_id_by_hash is None:
            if point_id_map_path.exists():
                try:
                    self._embedding_id_by_hash = json.load(open(point_id_map_path, "r"))
                except Exception:
                    self._embedding_id_by_hash = {}
            else:
                self._embedding_id_by_hash = {}
        
        point_id = self._embedding_id_by_hash.get(file_hash)
        if not point_id:
            point_id = str(uuid.uuid4())
            self._embedding_id_by_hash[file_hash] = point_id
            with open(point_id_map_path, "w") as f:
                json.dump(self._embedding_id_by_hash, f, indent=2, sort_keys=True)

        # Save embedding to file (for later Qdrant import)
        embedding_path = embeddings_dir / f"{file_hash}.npy"
        
        # If it already exists, load & return (do not append index again)
        if embedding_path.exists():
            embedding = np.load(embedding_path)
            if getattr(embedding, "ndim", 1) > 1:
                embedding = embedding[0]
            return embedding.tolist(), point_id

        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            embedding = outputs[0].cpu().numpy()
        
        np.save(embedding_path, embedding)
        
        # Also append to index file for batch import
        index_path = embeddings_dir / "index.jsonl"
        index_entry = {
            "point_id": point_id,
            "file_hash": file_hash,
            "image_path": str(image_path),
            "embedding_file": str(embedding_path),
            "dim": int(getattr(embedding, "shape", [len(embedding)])[-1]),
        }
        with open(index_path, 'a') as f:
            f.write(json.dumps(index_entry) + "\n")
        
        return embedding.tolist(), point_id
    
    def generate_caption(self, image_path: Path) -> Dict:
        """Generate caption and structured data using Qwen2.5-VL."""
        import torch
        
        qwen = self.model_manager.load_qwen()
        model = qwen["model"]
        processor = qwen["processor"]
        
        image = Image.open(image_path).convert("RGB")
        
        # Simple prompt - more reliable than complex structured prompts
        prompt = "Describe this image in one detailed sentence, mentioning the scene type, any people, and key objects."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
            )
        
        caption_text = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        
        # Extract nouns for DINO prompts
        extracted_nouns = self._extract_nouns(caption_text)
        
        result = {
            "text": caption_text,
            "short": caption_text[:self.config["max_caption_length"]],
            "extracted_nouns": extracted_nouns,
            "model": "qwen2.5-vl-3b",
        }
        
        return result
    
    def _extract_nouns(self, text: str) -> List[str]:
        """Extract localizable nouns from caption for GroundingDINO."""
        import re
        
        # Comprehensive stopwords including caption-specific terms
        stopwords = {
            # Common words
            "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "and", "or", "but",
            "this", "that", "these", "those", "it", "its", "their", "there",
            "here", "where", "when", "what", "which", "who", "how", "very",
            "some", "any", "all", "both", "each", "few", "more", "most",
            # Caption-specific (non-localizable)
            "image", "depicts", "shows", "features", "featuring", "scene",
            "photo", "photograph", "picture", "view", "shot", "capture",
            "background", "foreground", "setting", "atmosphere", "ambiance",
            "appears", "looks", "seems", "visible", "seen", "shown",
            # Adjectives commonly extracted as nouns
            "large", "small", "big", "little", "modern", "old", "new",
            "bright", "dark", "light", "warm", "cool", "cozy", "busy",
            "various", "several", "multiple", "many", "few", "single",
            "white", "black", "red", "blue", "green", "yellow", "brown",
            "wooden", "metal", "glass", "stone", "brick", "concrete",
            "indoor", "outdoor", "interior", "exterior",
            "left", "right", "center", "middle", "front", "back", "top", "bottom",
            # Verbs commonly extracted
            "sitting", "standing", "walking", "running", "eating", "drinking",
            "working", "reading", "looking", "wearing", "holding", "using",
            "covered", "surrounded", "filled", "placed", "located", "positioned",
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        nouns = [w for w in words if w not in stopwords]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_nouns = []
        for noun in nouns:
            if noun not in seen:
                seen.add(noun)
                unique_nouns.append(noun)
        
        return unique_nouns[:15]  # Limit to top 15
    
    def score_tags(self, image_path: Path, caption_data: Dict) -> List[Dict]:
        """Score image against taxonomy tags using SigLIP cosine similarity."""
        import torch
        import torch.nn.functional as F
        
        siglip = self.model_manager.load_siglip()
        model = siglip["model"]
        processor = siglip["processor"]
        
        image = Image.open(image_path).convert("RGB")
        
        # Get image embedding once
        img_inputs = processor(images=image, return_tensors="pt")
        img_inputs = {k: v.to(model.device) for k, v in img_inputs.items()}
        
        with torch.no_grad():
            img_embeds = model.get_image_features(**img_inputs)
            img_embeds = F.normalize(img_embeds, dim=-1)
        
        # Build/cache taxonomy text embeddings once (critical for scaling to large datasets)
        cache = self._taxonomy_prompt_cache
        if (
            cache is None
            or cache.get("device") != str(model.device)
            or cache.get("taxonomy_version") != getattr(self.taxonomy, "version", None)
        ):
            # Get all prompts from taxonomy
            all_prompts = self.taxonomy.get_all_prompts()
            
            # Flatten all prompts with their tag_ids
            prompt_list: List[str] = []
            prompt_to_tag: List[str] = []
            for tag_id, prompts in all_prompts.items():
                for prompt in prompts:
                    prompt_list.append(prompt)
                    prompt_to_tag.append(tag_id)
            
            if not prompt_list:
                return []
            
            chunk_size = 64
            txt_embed_chunks = []
            with torch.no_grad():
                for i in range(0, len(prompt_list), chunk_size):
                    chunk_prompts = prompt_list[i:i + chunk_size]
                    txt_inputs = processor(text=chunk_prompts, return_tensors="pt", padding=True)
                    txt_inputs = {k: v.to(model.device) for k, v in txt_inputs.items()}
                    txt_embeds = model.get_text_features(**txt_inputs)
                    txt_embeds = F.normalize(txt_embeds, dim=-1)
                    txt_embed_chunks.append(txt_embeds)
            
            txt_embeds_all = torch.cat(txt_embed_chunks, dim=0)
            
            self._taxonomy_prompt_cache = {
                "device": str(model.device),
                "taxonomy_version": getattr(self.taxonomy, "version", None),
                "prompt_list": prompt_list,
                "prompt_to_tag": prompt_to_tag,
                "txt_embeds": txt_embeds_all,
            }
            cache = self._taxonomy_prompt_cache
        
        prompt_list = cache["prompt_list"]
        prompt_to_tag = cache["prompt_to_tag"]
        txt_embeds_all = cache["txt_embeds"]
        
        with torch.no_grad():
            similarities = (img_embeds @ txt_embeds_all.T)[0]
            all_scores = similarities.detach().cpu().numpy().tolist()
        
        # Aggregate scores by tag_id (take max score per tag)
        tag_scores = {}
        tag_best_prompt = {}
        
        for prompt, tag_id, score in zip(prompt_list, prompt_to_tag, all_scores):
            if tag_id not in tag_scores or score > tag_scores[tag_id]:
                tag_scores[tag_id] = score
                tag_best_prompt[tag_id] = prompt
        
        # Convert to list and sort by score
        tag_list = [(tag_id, score) for tag_id, score in tag_scores.items()]
        tag_list.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k tags (relative ranking matters more than absolute threshold)
        top_k = int(self.config.get("top_k_tags", 10))
        verified_tags = []
        for tag_id, score in tag_list[:top_k]:
            tag_info = self.taxonomy.tag_info.get(tag_id, {})
            verified_tags.append({
                "tag_id": tag_id,
                "display": tag_info.get("display", tag_id),
                "category": tag_info.get("category", "unknown"),
                "confidence": round(float(score), 4),
                "source": "siglip_cosine",
                "evidence": {
                    "prompt_used": tag_best_prompt[tag_id],
                }
            })
        
        return verified_tags
    
    def detect_objects(self, image_path: Path, prompts: List[str]) -> List[Dict]:
        """Detect objects using GroundingDINO."""
        import torch
        
        if not prompts:
            return []
        
        dino = self.model_manager.load_grounding_dino()
        model = dino["model"]
        processor = dino["processor"]
        
        image = Image.open(image_path).convert("RGB")
        
        # Join prompts with periods (DINO format)
        text = ". ".join(prompts[:10]) + "."  # Limit to 10 prompts
        
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
        threshold = self.config["object_confidence_threshold"]
        mask = results["scores"] >= threshold
        scores = results["scores"][mask]
        labels = [results["labels"][i] for i, m in enumerate(mask) if m]
        boxes = results["boxes"][mask]
        
        objects = []
        for score, label, box in zip(scores, labels, boxes):
            box = box.cpu().numpy()
            
            # Normalize box coordinates (0-1)
            box_norm = {
                "x1": float(box[0] / image.width),
                "y1": float(box[1] / image.height),
                "x2": float(box[2] / image.width),
                "y2": float(box[3] / image.height),
            }
            
            # Pixel coordinates
            box_px = {
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
            }
            
            # Calculate area for z-order
            area = (box_px["x2"] - box_px["x1"]) * (box_px["y2"] - box_px["y1"])
            
            # Map to taxonomy
            label_canonical = self.taxonomy.map_phrase(label)
            
            objects.append({
                "object_id": str(uuid.uuid4()),
                "label": label,
                "label_canonical": label_canonical,
                "confidence": float(score),
                "box": box_norm,
                "box_pixels": box_px,
                "z_order": area,  # Smaller area = higher priority
            })
        
        # Sort by area (smaller first for click selection)
        objects.sort(key=lambda x: x["z_order"])
        
        # Assign z_order as rank
        for i, obj in enumerate(objects):
            obj["z_order"] = i
        
        return objects[:self.config["max_objects_per_image"]]
    
    def generate_masks(self, image_path: Path, objects: List[Dict]) -> List[Dict]:
        """Generate masks for detected objects using SAM2."""
        import torch
        
        if not objects:
            return objects
        
        sam2 = self.model_manager.load_sam2()
        if sam2 is None:
            console.print("[yellow]SAM2 not available, skipping masks[/yellow]")
            return objects
        
        model = sam2["model"]
        processor = sam2["processor"]
        
        image = Image.open(image_path).convert("RGB")
        
        # Process each object's bounding box
        for obj in objects:
            try:
                box = obj["box_pixels"]
                # SAM2 box format: [image_level, box_level, coords] = 3 levels
                input_boxes = [[[box["x1"], box["y1"], box["x2"], box["y2"]]]]
                
                inputs = processor(
                    images=image,
                    input_boxes=input_boxes,
                    return_tensors="pt"
                )
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Get best mask (highest IOU score)
                iou_scores = outputs.iou_scores[0, 0].cpu().numpy()
                best_idx = iou_scores.argmax()
                mask = outputs.pred_masks[0, 0, best_idx].cpu().numpy()
                
                # Threshold mask
                binary_mask = (mask > 0.5).astype('uint8')
                
                # Compute mask stats
                mask_area = int(binary_mask.sum())
                
                # Compute centroid
                if mask_area > 0:
                    y_coords, x_coords = binary_mask.nonzero()
                    centroid_x = float(x_coords.mean())
                    centroid_y = float(y_coords.mean())
                else:
                    centroid_x = (box["x1"] + box["x2"]) / 2
                    centroid_y = (box["y1"] + box["y2"]) / 2
                
                # RLE encode mask for storage
                mask_rle = self._rle_encode(binary_mask)
                
                # Update object with mask info
                obj["mask_rle"] = mask_rle
                obj["mask_area"] = mask_area
                obj["mask_iou"] = float(iou_scores[best_idx])
                obj["centroid"] = [centroid_x, centroid_y]
                
            except Exception as e:
                console.print(f"[yellow]Mask generation failed for {obj['label']}: {e}[/yellow]")
                obj["mask_rle"] = None
                obj["mask_area"] = 0
                obj["centroid"] = None
        
        return objects
    
    def _rle_encode(self, mask) -> str:
        """Run-length encode a binary mask."""
        import numpy as np
        
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        
        # Format: "start1 length1 start2 length2 ..."
        return ' '.join(str(x) for x in runs)
    
    def process_image(self, image_path: Path, source_type: str, source_category: str, 
                      lane: str = "all", output_dir: Path = None) -> ImageMetadata:
        """Process a single image through the pipeline."""
        
        # Basic info
        file_hash = self.compute_file_hash(image_path)
        img_info = self.get_image_info(image_path)
        
        # Try to load existing metadata (for Lane B updates)
        existing_metadata = None
        if output_dir and lane == "B":
            existing_metadata = self._find_existing_metadata(file_hash, output_dir)
        
        if existing_metadata:
            # Update existing metadata
            metadata = existing_metadata
            metadata.processed_at = datetime.now().isoformat()
        else:
            # Create new metadata
            metadata = ImageMetadata(
                image_id=str(uuid.uuid4()),
                file_hash=file_hash,
                file_path=str(image_path),
                width=img_info["width"],
                height=img_info["height"],
                source_type=source_type,
                source_category=source_category,
                processed_at=datetime.now().isoformat(),
            )
        
        # Lane A: Embedding
        if lane in ["A", "all"]:
            embedding, point_id = self.generate_embedding(image_path, file_hash)
            metadata.embedding_id = point_id
        
        # Lane A: Caption
        if lane in ["A", "all"]:
            caption = self.generate_caption(image_path)
            metadata.caption = caption
        
        # Lane A: Tags
        if lane in ["A", "all"]:
            tags = self.score_tags(image_path, metadata.caption or {})
            metadata.tags = tags
        
        # Lane B: Object Detection
        if lane in ["B", "all"]:
            # Start with localizable tags from taxonomy (high quality prompts)
            prompts = [
                self.taxonomy.tag_info[t]["display"] 
                for t in self.taxonomy.get_localizable_tags()
            ]
            
            # Add concrete nouns from caption (re-extract with current stopwords)
            if metadata.caption:
                caption_text = metadata.caption.get("text", "")
                if caption_text:
                    # Re-extract nouns with current stopwords
                    fresh_nouns = self._extract_nouns(caption_text)
                    for noun in fresh_nouns[:10]:
                        if noun not in prompts and len(noun) >= 3:
                            prompts.append(noun)
            
            objects = self.detect_objects(image_path, prompts)
            metadata.objects = objects
        
        # Lane B: Masks
        if lane in ["B", "all"] and metadata.objects:
            metadata.objects = self.generate_masks(image_path, metadata.objects)
        
        return metadata
    
    def _find_existing_metadata(self, file_hash: str, output_dir: Path) -> Optional[ImageMetadata]:
        """Find existing metadata by file hash."""
        if not output_dir.exists():
            return None
        
        for meta_file in output_dir.glob("*.json"):
            try:
                with open(meta_file, 'r') as f:
                    data = json.load(f)
                if data.get("file_hash") == file_hash:
                    # Convert dict back to ImageMetadata
                    return ImageMetadata(
                        image_id=data["image_id"],
                        file_hash=data["file_hash"],
                        file_path=data["file_path"],
                        width=data["width"],
                        height=data["height"],
                        source_type=data["source_type"],
                        source_category=data["source_category"],
                        caption=data.get("caption"),
                        tags=data.get("tags"),
                        objects=data.get("objects"),
                        embedding_id=data.get("embedding_id"),
                        processed_at=data.get("processed_at"),
                        processing_version=data.get("processing_version", "1.0.0"),
                    )
            except (json.JSONDecodeError, KeyError):
                continue
        return None
    
    def save_metadata(self, metadata: ImageMetadata, output_dir: Path):
        """Save metadata to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{metadata.image_id}.json"
        
        with open(output_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        return output_path


def ingest_images(input_dir: Path, output_file: Path):
    """Scan directory and create manifest of images to process."""
    console.print(f"\n[bold]Scanning {input_dir}...[/bold]")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = []
    
    for file_path in input_dir.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            # Determine category from path
            rel_path = file_path.relative_to(input_dir)
            category = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
            
            images.append({
                "path": str(file_path),
                "category": category,
                "size_bytes": file_path.stat().st_size,
            })
    
    console.print(f"[green]Found {len(images)} images[/green]")
    
    # Group by category
    categories = {}
    for img in images:
        cat = img["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    table = Table(title="Images by Category")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        table.add_row(cat, str(count))
    
    console.print(table)
    
    # Save manifest
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(images, f, indent=2)
    
    console.print(f"\n[green]Manifest saved to {output_file}[/green]")
    
    return images


def main():
    parser = argparse.ArgumentParser(description="Image Processing Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Scan and catalog images")
    ingest_parser.add_argument("--input-dir", type=Path, required=True)
    ingest_parser.add_argument(
        "--manifest", 
        type=Path, 
        default=Path("/data/image_pipeline/outputs/manifest.json")
    )
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process images")
    process_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/image_pipeline/outputs/manifest.json")
    )
    process_parser.add_argument(
        "--lane",
        choices=["A", "B", "all"],
        default="all",
        help="Processing lane (A=embed+caption+tag, B=detect+mask, all=everything)"
    )
    process_parser.add_argument("--limit", type=int, help="Limit number of images")
    process_parser.add_argument("--start", type=int, default=0, help="Start index")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test with single image")
    test_parser.add_argument("--image", type=Path, required=True)
    test_parser.add_argument("--lane", choices=["A", "B", "all"], default="all")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_images(args.input_dir, args.manifest)
        
    elif args.command == "process":
        # Load manifest
        with open(args.manifest, 'r') as f:
            images = json.load(f)
        
        # Apply limits
        if args.limit:
            images = images[args.start:args.start + args.limit]
        else:
            images = images[args.start:]
        
        console.print(f"\n[bold]Processing {len(images)} images (Lane {args.lane})[/bold]")
        
        pipeline = Pipeline(CONFIG)
        output_dir = CONFIG["output_dir"] / "metadata"
        
        # Track processed files for resume (lane-specific)
        lane_suffix = f"_lane_{args.lane.lower()}" if args.lane != "all" else ""
        processed_file = CONFIG["output_dir"] / f"processed_hashes{lane_suffix}.txt"
        processed_hashes = set()
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                processed_hashes = set(line.strip() for line in f if line.strip())
            console.print(f"[yellow]Resuming: {len(processed_hashes)} already processed[/yellow]")
        
        success_count = 0
        skip_count = 0
        error_count = 0
        
        for img_info in tqdm(images, desc="Processing"):
            try:
                # Quick hash check for resume
                img_path = Path(img_info["path"])
                file_hash = pipeline.compute_file_hash(img_path)
                
                if file_hash in processed_hashes:
                    skip_count += 1
                    continue
                
                metadata = pipeline.process_image(
                    img_path,
                    source_type="places365",
                    source_category=img_info["category"],
                    lane=args.lane,
                    output_dir=output_dir
                )
                pipeline.save_metadata(metadata, output_dir)
                
                # Track as processed
                with open(processed_file, 'a') as f:
                    f.write(file_hash + "\n")
                processed_hashes.add(file_hash)
                success_count += 1
                
            except Exception as e:
                console.print(f"[red]Error processing {img_info['path']}: {e}[/red]")
                error_count += 1
        
        console.print(f"\n[green]Processing complete![/green]")
        console.print(f"  Processed: {success_count}")
        console.print(f"  Skipped (already done): {skip_count}")
        console.print(f"  Errors: {error_count}")
        console.print(f"  Output: {output_dir}")
        
    elif args.command == "test":
        console.print(f"\n[bold]Testing pipeline on {args.image}[/bold]")
        
        pipeline = Pipeline(CONFIG)
        
        metadata = pipeline.process_image(
            args.image,
            source_type="test",
            source_category="test",
            lane=args.lane
        )
        
        # Pretty print result
        console.print("\n[bold]Result:[/bold]")
        console.print_json(json.dumps(asdict(metadata), indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
