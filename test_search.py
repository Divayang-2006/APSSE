import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import get_extractor, severity_analyzer

CATALOG_PATH = BACKEND_DIR / "dataset_cases.json"
CACHE_VIT_PATH = BACKEND_DIR / "embeddings_vit.npy"

def run_test_search(image_path: str = "", model_name: str = "vit", top_k: int = 4):
    target_path = None
    if image_path and Path(image_path).is_file():
        target_path = Path(image_path)
    else:
        # Default query image
        candidates = [
            BACKEND_DIR.parent / "dataset" / "extracted" / "Positive" / "00001.jpg",
            BACKEND_DIR / "static" / "samples" / "sample_query_1.jpg",
            BACKEND_DIR / "static" / "images" / "case_01.jpg"
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                target_path = c
                break

    if not target_path or not target_path.exists():
        print("Error: No query image found.")
        return

    # 1. Feature extraction for input image
    extractor = get_extractor(model_name)
    with Image.open(target_path) as img:
        img_rgb = img.convert("RGB")
        query_vec = extractor.extract(img_rgb)

    # 2. Fast similarity search against dataset
    matches = []
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)

        if CACHE_VIT_PATH.exists():
            matrix = np.load(CACHE_VIT_PATH)
            # Inner product of normalized vectors = Cosine similarity
            sims = np.dot(matrix[:len(cases)], query_vec)
            top_indices = np.argsort(sims)[::-1][:top_k]
            for idx in top_indices:
                matches.append({
                    "case_id": cases[idx]["case_id"],
                    "similarity": round(float(np.clip(sims[idx], 0.0, 1.0)), 2),
                    "title": cases[idx].get("title", ""),
                    "image_url": cases[idx].get("image_url", "")
                })
        else:
            scored = []
            for case in cases[:200]:
                img_file = case.get("local_file_path")
                if img_file and Path(img_file).exists():
                    try:
                        with Image.open(img_file) as c_img:
                            c_vec = extractor.extract(c_img.convert("RGB"))
                            sim = float(np.dot(query_vec, c_vec))
                            scored.append((case, sim))
                    except Exception:
                        continue
            scored.sort(key=lambda x: x[1], reverse=True)
            for case, sim in scored[:top_k]:
                matches.append({
                    "case_id": case["case_id"],
                    "similarity": round(float(np.clip(sim, 0.0, 1.0)), 2),
                    "title": case.get("title", ""),
                    "image_url": case.get("image_url", "")
                })

    # Exact required output format
    print(" Query Image")
    print(f"   [{target_path.resolve()}]")
    print("Similar Cases")
    print("Output should show image with score of similarity")
    for m in matches:
        cid = m["case_id"]
        sim = m["similarity"]
        print(f"{cid:<10}Similarity = {sim:.2f}")

    sev = severity_analyzer.analyze(img_rgb, top_cases=matches)
    print("----------------------------------------------------------------------")
    print(f"Automated Severity Assessment : {sev['score']:.2f} [{sev['level']}]")
    print(f"Structural Observations       : {', '.join(sev['key_observations'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate inspection image against dataset")
    parser.add_argument("--image", type=str, default="", help="Path to query image")
    parser.add_argument("--model", type=str, default="vit", help="Model backbone: vit or resnet50")
    parser.add_argument("--top-k", type=int, default=4, help="Number of similar cases to retrieve")
    args = parser.parse_args()

    run_test_search(image_path=args.image, model_name=args.model, top_k=args.top_k)
