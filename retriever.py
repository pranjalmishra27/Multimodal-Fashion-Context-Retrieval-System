import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# Common colors and garments to parse for compositional attribute matching
COLORS = {'yellow', 'red', 'blue', 'white', 'black', 'green', 'pink', 'brown', 'purple', 'orange', 'grey', 'gray', 'beige', 'colorful'}
GARMENTS = {'raincoat', 'tie', 'shirt', 'blazer', 'button-down', 'hoodie', 't-shirt', 'dress', 'pants', 'shorts', 'jacket', 'coat', 'suit', 'skirt', 'blouse', 'jeans', 'trousers', 'attire', 'outfit', 'poncho', 'cape', 'skirt', 'shorts', 'blanket'}

class GlanceRetriever:
    def __init__(self, index_dir=r"e:\Glance\index"):
        self.index_dir = index_dir
        self.metadata_path = os.path.join(index_dir, "index.json")
        self.embeddings_path = os.path.join(index_dir, "embeddings.npy")
        
        # Load index
        if not os.path.exists(self.metadata_path) or not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(f"Index files not found in {index_dir}. Please run the indexer first!")
            
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        self.images = self.metadata["images"]
        self.image_embeddings = np.load(self.embeddings_path)
        
        # Load models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Retriever] Loading CLIP and SentenceTransformer on {self.device}...")
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        
        # Precompute text embeddings for all VLM captions
        print("[Retriever] Precomputing caption text embeddings...")
        captions = [img["caption"] for img in self.images]
        self.caption_embeddings = self.text_model.encode(captions, convert_to_numpy=True, show_progress_bar=False)
        # Normalize
        self.caption_embeddings = self.caption_embeddings / np.linalg.norm(self.caption_embeddings, axis=-1, keepdims=True)
        
        print("[Retriever] Ready!")

    def parse_compositional_query(self, query):
        """
        Parses the query to extract (color, garment) pairs based on distance.
        E.g. 'a red tie and a white shirt' -> [('red', 'tie'), ('white', 'shirt')]
        """
        words = query.lower().replace('.', '').replace(',', '').split()
        color_indices = [i for i, w in enumerate(words) if w in COLORS]
        garment_indices = [i for i, w in enumerate(words) if w in GARMENTS]
        
        pairs = []
        for c_idx in color_indices:
            color = words[c_idx]
            # Find the closest garment index that comes AFTER the color (or closest in distance)
            best_g_idx = None
            min_dist = 999
            for g_idx in garment_indices:
                dist = g_idx - c_idx
                # Prefer garments that appear immediately after the color
                if dist > 0 and dist < min_dist:
                    min_dist = dist
                    best_g_idx = g_idx
            
            # If no garment is found after, just find the closest overall
            if best_g_idx is None:
                for g_idx in garment_indices:
                    dist = abs(g_idx - c_idx)
                    if dist < min_dist:
                        min_dist = dist
                        best_g_idx = g_idx
                        
            if best_g_idx is not None and min_dist <= 3:
                pairs.append((color, words[best_g_idx]))
                
        return pairs

    def check_compositional_match(self, caption, parsed_pairs):
        """
        Checks if the color-garment pairs are close to each other in the caption.
        Returns a score boost based on the fraction of matching pairs.
        """
        if not parsed_pairs:
            return 0.0
            
        caption_words = caption.lower().replace('.', '').replace(',', '').split()
        matches = 0
        
        for color, garment in parsed_pairs:
            # Find positions in caption
            c_indices = [i for i, w in enumerate(caption_words) if w == color]
            g_indices = [i for i, w in enumerate(caption_words) if w == garment]
            
            # Check if there is any pair close to each other (dist <= 3)
            matched_pair = False
            for ci in c_indices:
                for gi in g_indices:
                    if abs(gi - ci) <= 3:
                        matched_pair = True
                        break
                if matched_pair:
                    break
            if matched_pair:
                matches += 1
                
        # Return boost: +0.15 for each matched pair
        return 0.15 * matches

    def search(self, query, k=5, alpha=0.6):
        """
        Performs hybrid search on the indexed dataset.
        alpha: weight for CLIP visual score vs SentenceTransformer caption score.
        """
        # 1. Embed query with CLIP (text)
        clip_inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            query_clip = self.clip_model.get_text_features(**clip_inputs)
            query_clip = query_clip / query_clip.norm(dim=-1, keepdim=True)
            query_clip_np = query_clip.cpu().numpy()[0]
            
        # 2. Embed query with SentenceTransformer
        query_text_np = self.text_model.encode([query], convert_to_numpy=True)[0]
        query_text_np = query_text_np / np.linalg.norm(query_text_np)
        
        # 3. Calculate similarities
        # CLIP text-to-image similarity
        clip_scores = np.dot(self.image_embeddings, query_clip_np)
        
        # Text-to-text caption similarity
        text_scores = np.dot(self.caption_embeddings, query_text_np)
        
        # 4. Compute hybrid scores
        hybrid_scores = alpha * clip_scores + (1.0 - alpha) * text_scores
        
        # 5. Parse and apply compositional boost
        parsed_pairs = self.parse_compositional_query(query)
        boosts = []
        for img in self.images:
            boost = self.check_compositional_match(img["caption"], parsed_pairs)
            boosts.append(boost)
        boosts = np.array(boosts)
        
        final_scores = hybrid_scores + boosts
        
        # 6. Retrieve top-k indices
        top_indices = np.argsort(final_scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            img_data = self.images[idx]
            results.append({
                "rank": rank + 1,
                "filename": img_data["filename"],
                "dataset_index": img_data["dataset_index"],
                "caption": img_data["caption"],
                "clip_score": float(clip_scores[idx]),
                "text_score": float(text_scores[idx]),
                "composition_boost": float(boosts[idx]),
                "final_score": float(final_scores[idx])
            })
            
        return results, parsed_pairs
