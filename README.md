# Multimodal Fashion & Context Retrieval System

An intelligent search engine built for the Glance ML Internship Assignment. The system retrieves specific images from a fashion database based on natural language descriptions, understanding not just the garments worn but the surrounding environment, context, and overall vibe.

---

## Technical Highlights

### 1. Hybrid Retrieval Architecture
Standard CLIP (Zero-Shot Dual Encoder) is extremely fast but behaves like a bag-of-words model, frequently failing on compositionality (e.g. binding colors/textures to the wrong garments, like swapping "red tie and white shirt" with "red shirt and white tie"). 
To solve this, our system implements a **Dual-Encoder Hybrid Search**:
* **Visual Match (CLIP)**: Evaluates global text-to-image similarity using `openai/clip-vit-base-patch32`.
* **Relational VLM Captioning (BLIP)**: Offline auto-captions every image using `Salesforce/blip-image-captioning-base` to capture semantic associations.
* **Semantic Text-to-Text Match**: Computes cosine similarity between the query and captions using the SentenceTransformer `all-MiniLM-L6-v2`.

### 2. Compositional Attribute Proximity Boost
At query time, the system parses the user's text to identify `(color, garment)` bindings. If both the color and the garment appear close together (word distance $\le 3$) in the image's VLM description, the image receives a **+0.15 boost** per match. This filters out incorrect color swaps and ensures precise compositional retrieval.

### 3. Scalable & Zero-Shot Ready
* **Zero-Shot Handling**: The system leverages CLIP and BLIP's large-scale pre-training without label fine-tuning, allowing it to interpret completely novel styles, environments, and complex scenes.
* **1-Million Scale Strategy**: The indexing is linear O(N) and parallelizable, while online retrieval scales logarithmically O(log N) by storing embeddings in a production Vector Database (e.g. Qdrant or ChromaDB) utilizing Hierarchical Navigable Small World (HNSW) graphs.

---

## Repository Structure

```text
├── indexer.py                 # Part A: Offline feature extraction & VLM captioning
├── retriever.py               # Part B: Online hybrid retrieval engine & boost logic
├── search.py                  # Part B CLI: Search query interface
├── assignment_submission.pdf  # Final submission PDF report document
├── gallery.html               # Interactive glassmorphic search gallery of results
├── index/                     # Database storage folder
│   ├── embeddings.npy         # Precomputed CLIP image vectors
│   └── index.json             # Map of filenames, captions, and index mapping
└── test/                      # Raw JPEGs fashion dataset folder
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pranjalmishra27/Multimodal-Fashion-Context-Retrieval-System.git
   cd Multimodal-Fashion-Context-Retrieval-System
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision transformers sentence-transformers numpy pillow reportlab
   ```

---

## Usage

### 1. Run the Indexer (Part A)
Build the index database by extracting CLIP features and BLIP captions. The indexer supports progressive checkpointing to safely pause and resume progress.
```bash
python indexer.py --test-dir ./test --output-dir ./index --max-images 1000
```

### 2. Run the Retriever CLI (Part B)
Query the database using natural language. The CLI prints a formatted table of matching ranks, scores, and captions, and copies top visual matches to `search_results/` for inspection:
```bash
python search.py --query "A person in a bright yellow raincoat." -k 3
```

---

## Evaluation Prompts Verification

Our retriever was successfully verified against the five assignment prompts, achieving correct visual matches on the indexed database:

| Query | Top Match Filename | VLM Description | Proximity Boost |
| :--- | :--- | :--- | :--- |
| **1. Attribute Specific**: "A person in a bright yellow raincoat." | `e636280e96f3863157a4398c92fc299e.jpg` | "a little girl in a yellow raincoat and red tights" | `+0.15` (yellow raincoat) |
| **2. Contextual/Place**: "Professional business attire inside a modern office." | `ac9cf5f349291912f5a973202b51da55.jpg` | "a woman in a black suit and white shirt" | `0.00` |
| **3. Complex Semantic**: "Someone wearing a blue shirt sitting on a park bench." | `f06fd9e18c7a2c5b533e68de88fb487b.jpg` | "a young girl sitting on a bench in the park" | `0.00` |
| **4. Style Inference**: "Casual weekend outfit for a city walk." | `289236c117e289c12d6c57e2cb4ce427.jpg` | "a woman in a black coat and black shoes is walking down the street" | `0.00` |
| **5. Compositional**: "A red tie and a white shirt in a formal setting." | `05f4264fb2ef7d7928d8b85522672ec0.jpg` | "person in a black suit and red tie" | `+0.15` (red tie) |

*Note: In the baseline CLIP search, a zero-shot error placed `914d8db66c7d852b32df43fedfe1ed58.jpg` ("a woman sitting on a white box") as the top match for Query 5. Our hybrid retriever corrected this and successfully placed the actual red tie suit at rank #1.*
