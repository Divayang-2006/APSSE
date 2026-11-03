import os
import sys
import argparse
import json
import logging
from pathlib import Path
from PIL import Image
import numpy as np

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import get_extractor, severity_analyzer
from qdrant_store import qdrant_store, PointStruct

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_dataset")

DATASET_ROOT = BACKEND_DIR.parent / "dataset" / "extracted"
POSITIVE_DIR = DATASET_ROOT / "Positive"
NEGATIVE_DIR = DATASET_ROOT / "Negative"
OUTPUT_JSON = BACKEND_DIR / "dataset_cases.json"

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

def generate_case_metadata(img_path: Path, is_positive: bool, case_idx: int) -> dict:
    """Generates structural engineering metadata and automated severity scoring."""
    case_id = f"Case {case_idx:02d}"
    structure = STRUCTURES[case_idx % len(STRUCTURES)]
    location = LOCATIONS[(case_idx * 3) % len(LOCATIONS)]

    with Image.open(img_path) as pil_img:
        img_rgb = pil_img.convert("RGB")
        sev_info = severity_analyzer.analyze(img_rgb)

    if is_positive:
        score = max(0.55, min(0.96, sev_info["score"]))
        if score >= 0.85:
            level = "Critical"
            action = "Immediate structural shoring, high-pressure epoxy resin injection, and ultrasonic flaw depth profiling."
        elif score >= 0.70:
            level = "Severe"
            action = "Surface groove routing, elastomeric sealant application, and installation of crack monitoring displacement sensors."
        else:
            level = "Moderate"
            action = "Micro-silica mortar patching, water-repellent silane coating, and reinspection in 90 days."

        damage_type = "Concrete Structural Crack"
        desc = f"Linear surface crack observed on {structure}. Edge analysis confirms active tensile fracturing."
    else:
        score = max(0.05, min(0.30, sev_info["score"] * 0.4))
        level = "Minor"
        damage_type = "Intact Concrete Surface"
        action = "Surfaces sound. No immediate remedial action required. Maintain scheduled 2-year maintenance cycle."
        desc = f"Uniform monolithic concrete surface on {structure}. No visible spalling or fracture lines."

    rel_path = f"/dataset_images/{'Positive' if is_positive else 'Negative'}/{img_path.name}"

    return {
        "case_id": case_id,
        "title": f"{case_id}: {damage_type} ({structure})",
        "damage_type": damage_type,
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

def run_ingestion(total_samples: int = 1000, batch_size: int = 100, reset: bool = False):
    """Ingests dataset images into Qdrant and saves metadata catalog."""
    if not POSITIVE_DIR.exists() or not NEGATIVE_DIR.exists():
        logger.error(f"Dataset folders missing in {DATASET_ROOT}. Ensure extraction completed.")
        return

    pos_files = sorted(list(POSITIVE_DIR.glob("*.jpg")))
    neg_files = sorted(list(NEGATIVE_DIR.glob("*.jpg")))
    logger.info(f"Available in dataset: {len(pos_files)} Positive, {len(neg_files)} Negative images.")

    if total_samples <= 0 or total_samples > (len(pos_files) + len(neg_files)):
        total_samples = len(pos_files) + len(neg_files)

    half = total_samples // 2
    selected_pos = pos_files[:half]
    selected_neg = neg_files[:(total_samples - half)]
    combined = [(p, True) for p in selected_pos] + [(n, False) for n in selected_neg]

    logger.info(f"Target: Ingesting {len(combined)} images into Qdrant Vector DB (Positive: {len(selected_pos)}, Negative: {len(selected_neg)})...")

    # Ensure Qdrant collections
    qdrant_store.ensure_collection("damage_cases_vit", dimension=768)
    qdrant_store.ensure_collection("damage_cases_resnet", dimension=2048)

    vit_extractor = get_extractor("vit")
    resnet_extractor = get_extractor("resnet50")

    vit_points = []
    resnet_points = []
    catalog = []

    for idx, (img_path, is_pos) in enumerate(combined):
        try:
            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
                vit_vec = vit_extractor.extract(img_rgb).tolist()
                resnet_vec = resnet_extractor.extract(img_rgb).tolist()

            case_idx = idx + 1
            metadata = generate_case_metadata(img_path, is_pos, case_idx)
            catalog.append(metadata)

            vit_points.append(PointStruct(id=case_idx, vector=vit_vec, payload=metadata))
            resnet_points.append(PointStruct(id=case_idx, vector=resnet_vec, payload=metadata))

            if len(vit_points) >= batch_size:
                qdrant_store.upsert_batch("damage_cases_vit", vit_points, batch_size=batch_size)
                qdrant_store.upsert_batch("damage_cases_resnet", resnet_points, batch_size=batch_size)
                logger.info(f"Upserted {case_idx}/{len(combined)} points into Qdrant...")
                vit_points = []
                resnet_points = []

                # Periodic catalog flush
                if case_idx % 1000 == 0:
                    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                        json.dump(catalog, f, indent=2)

        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")

    # Upsert remaining points
    if vit_points:
        qdrant_store.upsert_batch("damage_cases_vit", vit_points)
        qdrant_store.upsert_batch("damage_cases_resnet", resnet_points)

    # Save final catalog JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    logger.info(f"Ingestion Finished! Total cases indexed in Qdrant: {len(catalog)}")
    logger.info(f"Saved catalog to {OUTPUT_JSON}")
    logger.info(f"Qdrant collection 'damage_cases_vit' count: {qdrant_store.count('damage_cases_vit')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Concrete Crack Dataset into Qdrant Vector DB")
    parser.add_argument("--samples", type=int, default=1000, help="Number of images to ingest (0 for all 40,000, default: 1000)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for upserting into Qdrant")
    args = parser.parse_args()

    run_ingestion(total_samples=args.samples, batch_size=args.batch_size)
