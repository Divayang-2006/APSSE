import io
import time
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from database import CASES_METADATA, STATIC_DIR, IMAGE_DIR, SAMPLE_DIR
from models import get_extractor, severity_analyzer, FeatureExtractor
from vector_search import VectorIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("apsse-api")

app = FastAPI(
    title="AI-Powered Damage Image Similarity Search Engine",
    description="Vector similarity search for structural damage inspection using ViT & ResNet embeddings.",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path
from fastapi.responses import FileResponse

# Mount static files for images
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

# Build Vector Indexes for both ViT (768-d) and ResNet (2048-d)
vit_index = VectorIndex(dimension=768)
resnet_index = VectorIndex(dimension=2048)

case_lookup = {case["case_id"]: case for case in CASES_METADATA}

def index_all_cases():
    logger.info("Generating embeddings and building vector indexes...")
    vit_ext = get_extractor("vit")
    res_ext = get_extractor("resnet50")

    for case in CASES_METADATA:
        img_path = IMAGE_DIR / f"{case['case_id'].replace(' ', '_').lower()}.jpg"
        # fallback to case_01.jpg etc
        idx = CASES_METADATA.index(case) + 1
        img_path = IMAGE_DIR / f"case_{idx:02d}.jpg"
        
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            # ViT embedding
            v_embed = vit_ext.extract(img)
            vit_index.add(case["case_id"], v_embed)

            # ResNet embedding
            r_embed = res_ext.extract(img)
            resnet_index.add(case["case_id"], r_embed)

    logger.info(f"Indexed {vit_index.count()} cases into HNSW / Cosine indexes.")

# Initialize vector stores
index_all_cases()

@app.get("/")
def root():
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "system": "AI-Powered Damage Image Similarity Search Engine",
        "status": "Online",
        "docs_url": "/docs",
        "total_cases_indexed": len(CASES_METADATA)
    }

@app.post("/api/search")
async def search_similar_damage(
    file: UploadFile = File(...),
    model: str = Form("vit"),
    top_k: int = Form(4)
):
    """
    1. Receives Query Image
    2. Feature Extraction (ViT / ResNet)
    3. Normalizes to unit vector
    4. ANN Similarity Search
    5. Returns Top-K Similar Cases with scores & Severity Analysis
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

    # 2. ANN Vector Search
    index = resnet_index if "resnet" in selected_model else vit_index
    search_hits = index.search(query_vector, top_k=top_k)

    # 3. Format Matched Cases
    results = []
    for case_id, score in search_hits:
        c = case_lookup.get(case_id, {})
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

    # 4. Severity Analysis
    severity = severity_analyzer.analyze(query_image, top_cases=results)

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "status": "success",
        "latency_ms": latency_ms,
        "model_used": "Vision Transformer (ViT-B/16)" if "vit" in selected_model else "ResNet-50",
        "vector_dimension": extractor.dim,
        "similarity_metric": "Cosine Similarity (MIPS)",
        "severity_analysis": severity,
        "results": results
    }

@app.get("/api/cases")
def list_cases(damage_type: Optional[str] = None):
    """Browse historical inspection cases catalog."""
    if damage_type:
        filtered = [c for c in CASES_METADATA if damage_type.lower() in c["damage_type"].lower()]
        return {"total": len(filtered), "cases": filtered}
    return {"total": len(CASES_METADATA), "cases": CASES_METADATA}

@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    case = case_lookup.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return case

@app.get("/api/samples")
def get_sample_images():
    """Provides sample inspection images for one-click instant testing."""
    return [
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

@app.get("/api/stats")
def get_stats():
    """Returns vector database index status and cluster metrics."""
    damage_dist = {}
    severity_dist = {}
    for c in CASES_METADATA:
        dt = c.get("damage_type", "Other")
        sl = c.get("severity_level", "Other")
        damage_dist[dt] = damage_dist.get(dt, 0) + 1
        severity_dist[sl] = severity_dist.get(sl, 0) + 1

    return {
        "total_cases": len(CASES_METADATA),
        "vit_index_size": vit_index.count(),
        "resnet_index_size": resnet_index.count(),
        "vector_dimensions": {
            "vit": 768,
            "resnet50": 2048
        },
        "index_algorithm": "Hierarchical Navigable Small World (HNSW) + Cosine Distance",
        "damage_type_distribution": damage_dist,
        "severity_distribution": severity_dist,
        "cluster_node": "node-worker-apse1",
        "container_status": "healthy"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
