import os, sys, json, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\backend")
import models

DATASET_ROOT = Path(r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\dataset\extracted")
POS_DIR = DATASET_ROOT / "Positive"
NEG_DIR = DATASET_ROOT / "Negative"
BACKEND_DIR = Path(r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\backend")
OUT_JSON = BACKEND_DIR / "dataset_cases.json"
OUT_NPY = BACKEND_DIR / "embeddings_vit.npy"

STRUCTURES = [
    "Bridge Pier Support Span B-4",
    "Highway Overpass Girder G-12",
    "Prestressed Concrete Viaduct Deck",
    "Foundation Retaining Wall Node 7",
    "Subway Tunnel Invert Slab",
    "Seismic Retrofit Abutment Pier",
    "Reinforced Concrete Shear Wall",
    "Water Treatment Basalt Tank Wall"
]

LOCATIONS = [
    "Interstate 80 Highway Milepost 44",
    "Metro Rapid Transit Elevated Sector 2",
    "Hudson River Bridge Approach Span",
    "Industrial Port Terminal Berth 14",
    "Olympic Stadium Overpass Segment C",
    "Mountain Highway Viaduct Pier 9"
]

def compute_case_metadata(img_path: Path, is_positive: bool, case_idx: int) -> dict:
    case_id = f"Case {case_idx:02d}"
    structure = STRUCTURES[case_idx % len(STRUCTURES)]
    location = LOCATIONS[(case_idx * 3) % len(LOCATIONS)]

    with Image.open(img_path) as pil_img:
        img_rgb = pil_img.convert("RGB")
        sev_info = models.severity_analyzer.analyze(img_rgb)

    score = sev_info["score"]
    level = sev_info["level"]
    dtype = sev_info["damage_type"]

    if is_positive:
        dtype = "Concrete Structural Crack"
        if score >= 0.80:
            level = "Critical"
            action = "Immediate structural shoring, high-pressure epoxy resin injection, and ultrasonic flaw depth profiling."
            desc = f"Active high-propagation tensile crack observed on {structure}. Immediate remediation required."
        elif score >= 0.60:
            level = "Severe"
            action = "Surface groove routing, elastomeric sealant application, and installation of crack monitoring sensors."
            desc = f"Significant structural fracture line on {structure}. Crack width exceeds standard durability limits."
        else:
            level = "Moderate"
            action = "Micro-silica mortar patching, water-repellent silane coating, and reinspection in 90 days."
            desc = f"Surface crack observed on {structure}. Edge analysis confirms stable superficial fracturing."
    else:
        dtype = "Intact Concrete Surface"
        score = min(0.20, score)
        level = "Minor"
        action = "Surfaces monolithic and sound. Maintain standard 2-year maintenance cycle."
        desc = f"Uniform monolithic concrete surface on {structure}. No visible spalling or fracture lines."

    rel_path = f"/dataset_images/{'Positive' if is_positive else 'Negative'}/{img_path.name}"
    return {
        "case_id": case_id,
        "title": f"{case_id}: {dtype} ({structure})",
        "damage_type": dtype,
        "severity_level": level,
        "severity_score": round(score, 2),
        "structure_type": structure,
        "inspection_date": f"2026-{(case_idx % 12) + 1:02d}-{(case_idx % 28) + 1:02d}",
        "location": location,
        "description": desc,
        "recommended_action": action,
        "image_url": rel_path,
        "local_file_path": str(img_path)
    }

def main():
    print("Initializing Real ViT feature extractor...")
    models._FORCE_FALLBACK = False
    models.TORCH_AVAILABLE = True
    extractor = models.FeatureExtractor("vit")
    if extractor.model is None:
        print("ERROR: Could not load real ViT model!")
        return

    # Select 600 Positive + 400 Negative images (1,000 balanced catalog)
    pos_files = sorted(list(POS_DIR.glob("*.jpg")))[:600]
    neg_files = sorted(list(NEG_DIR.glob("*.jpg")))[:400]
    
    all_files = [(p, True) for p in pos_files] + [(n, False) for n in neg_files]
    print(f"Target catalog: {len(all_files)} cases (Positive: {len(pos_files)}, Negative: {len(neg_files)})")

    # Check if we can reuse existing 600 positive embeddings to save time
    existing_emb = None
    if OUT_NPY.exists():
        existing_emb = np.load(str(OUT_NPY))
        print(f"Loaded existing embeddings matrix: {existing_emb.shape}")

    catalog = []
    embeddings = []

    start = time.time()
    for idx, (img_path, is_pos) in enumerate(all_files):
        case_idx = idx + 1
        meta = compute_case_metadata(img_path, is_pos, case_idx)
        catalog.append(meta)

        # If already computed in existing_emb and is positive in first 600
        if is_pos and existing_emb is not None and idx < len(existing_emb):
            vec = existing_emb[idx]
        else:
            with Image.open(img_path) as pil_img:
                vec = extractor.extract(pil_img.convert("RGB"))
        embeddings.append(vec)

        if case_idx % 100 == 0:
            print(f"Processed {case_idx}/{len(all_files)} cases ({time.time() - start:.1f}s)...")

    emb_matrix = np.array(embeddings, dtype=np.float32)
    print(f"Saving catalog to {OUT_JSON} ({len(catalog)} entries)...")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"Saving embeddings to {OUT_NPY} ({emb_matrix.shape})...")
    np.save(str(OUT_NPY), emb_matrix)

    # Show distribution
    scores = [c["severity_score"] for c in catalog]
    levels = [c["severity_level"] for c in catalog]
    types = [c["damage_type"] for c in catalog]
    print(f"Done! Severity min={min(scores)}, max={max(scores)}, mean={np.mean(scores):.2f}")
    print(f"Levels: Critical={levels.count('Critical')}, Severe={levels.count('Severe')}, Moderate={levels.count('Moderate')}, Minor={levels.count('Minor')}")
    print(f"Types: Crack={types.count('Concrete Structural Crack')}, Intact={types.count('Intact Concrete Surface')}")

if __name__ == "__main__":
    main()

