import io
import sys
import time
import logging
from typing import Optional
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

from database import CASES_METADATA, STATIC_DIR, IMAGE_DIR, SAMPLE_DIR, get_combined_catalog
from models import get_extractor, severity_analyzer, FeatureExtractor
from vector_search import VectorIndex
from qdrant_store import qdrant_store, PointStruct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("apsse-api")

app = FastAPI(
    title="AI-Powered Damage Image Similarity Search Engine",
    description="Vector similarity search for structural damage inspection using ViT, ResNet & Qdrant Vector DB.",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for default synthetic images & samples
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount real dataset images if extracted directory exists
DATASET_EXTRACTED_DIR = Path(__file__).resolve().parent.parent / "dataset" / "extracted"
if DATASET_EXTRACTED_DIR.exists():
    app.mount("/dataset_images", StaticFiles(directory=str(DATASET_EXTRACTED_DIR)), name="dataset_images")
    logger.info(f"Mounted real concrete dataset images from {DATASET_EXTRACTED_DIR}")

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

# Build Vector Indexes (Fallback FAISS / in-memory MIPS)
vit_index = VectorIndex(dimension=768)
resnet_index = VectorIndex(dimension=2048)

# Initialize Qdrant collections
qdrant_store.ensure_collection("damage_cases_vit", dimension=768)
qdrant_store.ensure_collection("damage_cases_resnet", dimension=2048)

def get_catalog_map():
    return {c["case_id"]: c for c in get_combined_catalog()}

def index_initial_cases():
    """Seeds in-memory FAISS and Qdrant with base industrial damage catalog."""
    logger.info("Verifying initial cases in vector stores...")
    vit_ext = get_extractor("vit")
    res_ext = get_extractor("resnet50")

    vit_points = []
    resnet_points = []

    for idx, case in enumerate(CASES_METADATA):
        img_path = IMAGE_DIR / f"case_{idx+1:02d}.jpg"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            v_embed = vit_ext.extract(img)
            r_embed = res_ext.extract(img)

            # In-memory FAISS
            vit_index.add(case["case_id"], v_embed)
            resnet_index.add(case["case_id"], r_embed)

            # Prepare Qdrant seed points (if not already seeded)
            if qdrant_store.count("damage_cases_vit") < len(CASES_METADATA):
                p_id = 90000 + idx
                vit_points.append(PointStruct(id=p_id, vector=v_embed.tolist(), payload=case))
                resnet_points.append(PointStruct(id=p_id, vector=r_embed.tolist(), payload=case))

    if vit_points:
        qdrant_store.upsert_batch("damage_cases_vit", vit_points)
        qdrant_store.upsert_batch("damage_cases_resnet", resnet_points)

    logger.info(f"Vector stores ready. In-memory FAISS count: {vit_index.count()}, Qdrant ViT count: {qdrant_store.count('damage_cases_vit')}")

# Run startup index check
index_initial_cases()

@app.get("/")
def root():
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "system": "AI-Powered Damage Image Similarity Search Engine",
        "status": "Online",
        "vector_db": "Qdrant",
        "qdrant_mode": "Cluster Server" if qdrant_store.use_server else "Embedded Persistent",
        "total_cases_indexed": qdrant_store.count("damage_cases_vit") or len(get_combined_catalog()),
        "docs_url": "/docs"
    }

@app.post("/api/search")
async def search_similar_damage(
    file: UploadFile = File(...),
    model: str = Form("vit"),
    top_k: int = Form(4)
):
    """
    1. Receives Query Image
    2. Deep Feature Extraction (ViT / ResNet)
    3. Normalizes to unit vector
    4. Standalone Vector DB ANN Search (Qdrant with Cosine Metric)
    5. Returns Top-K Similar Cases with scores & Consensus Severity Analysis
    """
    start_time = time.time()
    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # 1. Feature Extraction
    selected_model = model.lower()
    extractor = get_extractor(selected_model)
    query_vector = extractor.extract(query_image)

    # Collection selection
    col_name = "damage_cases_resnet" if "resnet" in selected_model else "damage_cases_vit"
    results = []

    # 2. ANN Search via Standalone Vector DB (Qdrant)
    qdrant_hits = qdrant_store.search(col_name, query_vector, top_k=top_k)
    if qdrant_hits:
        for hit in qdrant_hits:
            results.append({
                "case_id": hit.get("case_id", "Unknown"),
                "title": hit.get("title", ""),
                "similarity": hit.get("similarity", 0.0),
                "damage_type": hit.get("damage_type", "Structural Defect"),
                "severity_level": hit.get("severity_level", "Moderate"),
                "severity_score": hit.get("severity_score", 0.7),
                "structure_type": hit.get("structure_type", "Concrete Structure"),
                "inspection_date": hit.get("inspection_date", "2026-01-01"),
                "location": hit.get("location", ""),
                "description": hit.get("description", ""),
                "recommended_action": hit.get("recommended_action", ""),
                "image_url": hit.get("image_url", "")
            })
    else:
        # Fallback to FAISS / In-Memory matrix if Qdrant has no items
        index = resnet_index if "resnet" in selected_model else vit_index
        search_hits = index.search(query_vector, top_k=top_k)
        lookup = get_catalog_map()
        for case_id, score in search_hits:
            c = lookup.get(case_id, {})
            results.append({
                "case_id": case_id,
                "title": c.get("title", ""),
                "similarity": score,
                "damage_type": c.get("damage_type", "Structural Defect"),
                "severity_level": c.get("severity_level", "Moderate"),
                "severity_score": c.get("severity_score", 0.7),
                "structure_type": c.get("structure_type", "Industrial Structure"),
                "inspection_date": c.get("inspection_date", "2026-01-01"),
                "location": c.get("location", ""),
                "description": c.get("description", ""),
                "recommended_action": c.get("recommended_action", ""),
                "image_url": c.get("image_url", "")
            })

    # 3. Severity Analysis & Multi-match Consensus
    severity = severity_analyzer.analyze(query_image, top_cases=results)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "status": "success",
        "latency_ms": latency_ms,
        "model_used": "Vision Transformer (ViT-B/16)" if "vit" in selected_model else "ResNet-50",
        "vector_dimension": extractor.dim,
        "vector_database": "Qdrant" if qdrant_hits else "FAISS (In-Memory Fallback)",
        "similarity_metric": "Cosine Similarity (MIPS)",
        "severity_analysis": severity,
        "results": results
    }

@app.get("/api/cases")
def list_cases(damage_type: Optional[str] = None, limit: int = 50):
    """Browse historical inspection cases catalog."""
    catalog = get_combined_catalog()
    if damage_type:
        filtered = [c for c in catalog if damage_type.lower() in c.get("damage_type", "").lower()]
        return {"total": len(filtered), "cases": filtered[:limit]}
    return {"total": len(catalog), "cases": catalog[:limit]}

@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    lookup = get_catalog_map()
    case = lookup.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return case

@app.get("/api/samples")
def get_sample_images():
    """Provides sample inspection images for one-click instant testing."""
    samples = [
        {
            "id": 1,
            "name": "Fatigue Crack in Steel Girder",
            "url": "/static/samples/sample_query_1.jpg",
            "expected_match": "Case 01",
            "damage_type": "Fatigue Crack"
        },
        {
            "id": 2,
            "name": "Pitting Corrosion on Offshore Leg",
            "url": "/static/samples/sample_query_2.jpg",
            "expected_match": "Case 18",
            "damage_type": "Pitting Corrosion"
        },
        {
            "id": 3,
            "name": "Concrete Rebar Spalling",
            "url": "/static/samples/sample_query_3.jpg",
            "expected_match": "Case 42",
            "damage_type": "Concrete Spalling"
        }
    ]

    # Add real concrete crack sample if extracted
    if (DATASET_EXTRACTED_DIR / "Positive" / "00001.jpg").exists():
        samples.append({
            "id": 4,
            "name": "Real Concrete Fracture (Kaggle Dataset)",
            "url": "/dataset_images/Positive/00001.jpg",
            "expected_match": "CONC_POS_00001",
            "damage_type": "Concrete Structural Crack"
        })
    return samples

@app.get("/api/stats")
def get_stats():
    """Returns vector database index status and cluster metrics."""
    catalog = get_combined_catalog()
    damage_dist = {}
    severity_dist = {}
    for c in catalog:
        dt = c.get("damage_type", "Other")
        sl = c.get("severity_level", "Other")
        damage_dist[dt] = damage_dist.get(dt, 0) + 1
        severity_dist[sl] = severity_dist.get(sl, 0) + 1

    vit_qdrant_count = qdrant_store.count("damage_cases_vit")
    res_qdrant_count = qdrant_store.count("damage_cases_resnet")

    return {
        "total_cases_catalog": len(catalog),
        "vector_database": "Qdrant (Standalone Vector DB)",
        "qdrant_storage_mode": "Cluster Server" if qdrant_store.use_server else "Embedded Persistent Disk",
        "vit_index_size": vit_qdrant_count or vit_index.count(),
        "resnet_index_size": res_qdrant_count or resnet_index.count(),
        "vector_dimensions": {
            "vit": 768,
            "resnet50": 2048
        },
        "index_algorithm": "HNSW Graph Index + Cosine Distance",
        "damage_type_distribution": damage_dist,
        "severity_distribution": severity_dist,
        "cluster_node": "node-worker-apse1",
        "container_status": "healthy"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
