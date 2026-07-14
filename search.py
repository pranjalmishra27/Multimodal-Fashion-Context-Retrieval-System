import os
import argparse
import shutil
from retriever import GlanceRetriever

def main():
    parser = argparse.ArgumentParser(description="Glance Fashion Retriever (Part B)")
    parser.add_argument("--query", type=str, required=True, help="Natural language search query")
    parser.add_argument("-k", type=int, default=5, help="Number of top matching images to retrieve")
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for CLIP visual score (0.0 to 1.0)")
    parser.add_argument("--index-dir", type=str, default=r"e:\Glance\index", help="Index directory")
    parser.add_argument("--output-dir", type=str, default=r"e:\Glance\search_results", help="Directory to save copy of retrieved images")
    parser.add_argument("--test-dir", type=str, default=r"e:\Glance\test", help="Directory containing raw images")
    args = parser.parse_args()

    # Initialize retriever
    try:
        retriever = GlanceRetriever(index_dir=args.index_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run indexer.py first to build the index!")
        return

    # Perform search
    print(f"\nSearching for: '{args.query}' (k={args.k}, alpha={args.alpha})...")
    results, parsed_pairs = retriever.search(args.query, k=args.k, alpha=args.alpha)
    
    if parsed_pairs:
        print(f"Parsed color-garment bindings: {parsed_pairs}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Clean previous results in output dir
    for f in os.listdir(args.output_dir):
        fp = os.path.join(args.output_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

    # Print results in markdown format and copy images
    print("\n### Top Matching Images")
    print("| Rank | Filename | Final Score | CLIP Score | Caption Text Score | Boost | VLM Caption |")
    print("|------|----------|-------------|------------|-------------------|-------|-------------|")
    
    for r in results:
        # Copy image to output folder
        src_path = os.path.join(args.test_dir, r["filename"])
        dest_filename = f"rank_{r['rank']}_{r['filename']}"
        dest_path = os.path.join(args.output_dir, dest_filename)
        
        copied_status = "Error"
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dest_path)
                copied_status = "Copied"
            except Exception as e:
                copied_status = f"Copy Error ({e})"
        else:
            copied_status = "Not Found"
            
        print(f"| {r['rank']} | {r['filename']} | {r['final_score']:.4f} | {r['clip_score']:.4f} | {r['text_score']:.4f} | {r['composition_boost']:.4f} | {r['caption']} |")

    print(f"\nTop-{args.k} images copied to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
