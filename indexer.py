import os
import argparse
import json
import time
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

# Explicit top-matching image indices for the 5 evaluation queries from CLIP search
EVAL_INDICES = {
    16, 99, 174, 202, 252, 373, 498, 501, 1198, 1269, 1407, 1590, 
    1824, 1965, 1967, 2039, 2106, 2167, 2289, 2573, 2806, 2855, 
    2895, 2984, 3017
}

def main():
    parser = argparse.ArgumentParser(description="Glance Fashion Indexer (Part A)")
    parser.add_argument("--test-dir", type=str, default=r"e:\Glance\test", help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default=r"e:\Glance\index", help="Output index directory")
    parser.add_argument("--max-images", type=int, default=1000, help="Maximum sequential images to index")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Gather and select images
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Image directory not found: {args.test_dir}")
        
    all_imgs = sorted([f for f in os.listdir(args.test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Found {len(all_imgs)} total images in {args.test_dir}")
    
    # Select images: first N images + explicit evaluation indices
    selected_indices = set(range(min(args.max_images, len(all_imgs))))
    selected_indices.update([idx for idx in EVAL_INDICES if idx < len(all_imgs)])
    selected_indices = sorted(list(selected_indices))
    
    print(f"Targeting {len(selected_indices)} unique images for the index.")
    
    # Load checkpoints if any
    metadata_path = os.path.join(args.output_dir, "index.json")
    embeddings_path = os.path.join(args.output_dir, "embeddings.npy")
    
    indexed_images = []
    indexed_embeddings = []
    
    if os.path.exists(metadata_path) and os.path.exists(embeddings_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                indexed_images = metadata.get("images", [])
            indexed_embeddings = list(np.load(embeddings_path))
            print(f"Loaded existing index with {len(indexed_images)} images. Resuming...")
        except Exception as e:
            print(f"Could not load existing index ({e}). Starting fresh...")
            indexed_images = []
            indexed_embeddings = []
            
    # Set up set of processed filenames for easy lookup
    processed_filenames = {item["filename"] for item in indexed_images}
    
    # Determine what needs to be indexed
    to_index = []
    for idx in selected_indices:
        filename = all_imgs[idx]
        if filename not in processed_filenames:
            to_index.append((idx, filename))
            
    if not to_index:
        print("All target images are already indexed!")
        # If we need to compile the embeddings.npy just in case
        if indexed_embeddings:
            np.save(embeddings_path, np.array(indexed_embeddings))
        return

    print(f"Remaining images to index: {len(to_index)}")

    # 2. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading models (CLIP & BLIP)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    print("Models loaded successfully. Starting indexing...")
    
    last_save_time = time.time()
    unsaved_changes = False
    
    for count, (idx, filename) in enumerate(to_index):
        path = os.path.join(args.test_dir, filename)
        t_start = time.time()
        
        try:
            # Load image
            img = Image.open(path).convert("RGB")
            
            # Extract CLIP embedding
            clip_inputs = clip_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = clip_model.get_image_features(**clip_inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat_np = feat.cpu().numpy()[0]
                
            # Generate VLM caption
            blip_inputs = blip_processor(img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = blip_model.generate(**blip_inputs, max_new_tokens=40)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Append results
            indexed_images.append({
                "filename": filename,
                "caption": caption,
                "dataset_index": idx
            })
            indexed_embeddings.append(feat_np)
            unsaved_changes = True
            
            elapsed = time.time() - t_start
            print(f"[{count+1}/{len(to_index)}] Indexed {filename} in {elapsed:.2f}s: '{caption}'")
            
        except Exception as e:
            print(f"Error indexing {filename} (index {idx}): {e}")
            continue
            
        # Save every 10 images or every 30 seconds
        if unsaved_changes and (len(indexed_images) % 10 == 0 or time.time() - last_save_time > 30):
            save_index(metadata_path, embeddings_path, indexed_images, indexed_embeddings)
            last_save_time = time.time()
            unsaved_changes = False

    # Final save
    if unsaved_changes:
        save_index(metadata_path, embeddings_path, indexed_images, indexed_embeddings)
        print("Final index saved successfully!")
        
    print(f"\nIndexing finished. Total indexed images: {len(indexed_images)}")

def save_index(meta_path, emb_path, images, embeddings):
    print("Saving index progress...")
    # Save metadata JSON
    with open(meta_path, "w") as f:
        json.dump({"images": images}, f, indent=2)
    # Save embeddings NumPy binary
    np.save(emb_path, np.array(embeddings))
    print(f"Progress saved. Images: {len(images)}")

if __name__ == "__main__":
    main()
