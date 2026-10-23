# AI-Powered Damage Image Similarity Search Engine

> **Visual Search & Severity Analysis for Structural Inspection**  
> Retrieve similar historical damage cases (cracks, corrosion, spalling, weld defects) using dense visual embeddings rather than filenames or keywords.

---

## 📌 System Architecture

```
[ Engineer / Browser ]
        │  (Upload Query Image)
        ▼
[ React Dashboard ] ──(HTTP/REST)──▶ [ FastAPI Service ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ Feature Extraction ]                      [ Severity Estimator ]
             - Model: ViT / ResNet                       - Defect Classifier
             - L2-Normalized Embedding                   - k-NN Consensus Fusion
                       │
                       ▼
             [ Vector Search Engine ]
             - HNSW Graph Index (FAISS)
             - Metric: Cosine Similarity (MIPS)
             - Top-K IDs + Similarity Scores
                       │
                       ▼
             [ Case Metadata Catalog ]
             - Detailed historical damage records
             - Recommended remediation actions
```

---

## 🧠 Algorithms & Models Used

1. **Feature Extraction Backbones**:
   - **Vision Transformer (ViT-B/16)**: Extracts self-attention patch features into a **768-dimensional** dense vector.
   - **ResNet-50**: CNN backbone using Global Average Pooling (GAP) from the penultimate layer producing a **2048-dimensional** dense vector.
   - **$L_2$ Normalization**: Each embedding $v$ is normalized such that $\|v\|_2 = 1$.

2. **Similarity Metric**:
   - **Cosine Similarity / Maximum Inner Product Search (MIPS)**:
     $$\text{Cosine Sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v \quad (\text{for unit vectors})$$
   - Computes exact match score between $0.0$ and $1.0$.

3. **Approximate Nearest Neighbor (ANN) Indexing**:
   - **HNSW (Hierarchical Navigable Small World)** via FAISS: Multi-layer graph traversal with $M=16$, $efConstruction=200$, and $efSearch=64$ for sub-millisecond retrieval with $>98\%$ recall.

4. **Severity Analysis & Consensus**:
   - Combines image edge frequency, rust oxidation masks, and **distance-weighted k-NN consensus** from historical matches to rate damage severity (Minor, Moderate, Severe, Critical).

---

## 📂 Project Structure (Simple & Clean)

```
APSSE/
├── backend/
│   ├── main.py             # FastAPI REST API & routes (/api/search, /api/cases)
│   ├── models.py           # ViT, ResNet-50 extractors & Severity Analyzer
│   ├── vector_search.py    # HNSW Index & Cosine Similarity search engine
│   ├── database.py         # Historical damage catalog & sample inspection images
│   ├── requirements.txt    # Python dependencies
│   └── static/             # Storage for inspection images
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # React inspection dashboard with severity analysis
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   └── dist/               # Pre-built production bundle
├── Dockerfile              # Container image for cluster deployment
├── docker-compose.yml      # Orchestration for distributed software challenge
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Run with Python (Backend + UI in 1 Command)
The backend automatically serves both the REST API and the pre-built React UI:
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Start engine
python backend/main.py
```
Open your browser at **[http://localhost:8000](http://localhost:8000)**.

### 2. Frontend Development Mode (Optional)
If you want live hot-reloading for React:
```bash
cd frontend
npm install
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)**.

### 3. Run with Docker (Distributed Container Deployment)
```bash
docker-compose up --build
```

---

## 📊 Sample Output (Matching Project Requirements)

```
Query Image: Transverse crack on bridge pier
Model: Vision Transformer (ViT-B/16)
Severity Analysis: 0.94 (Critical) - High continuous fracture risk
Retrieval Latency: 14.8 ms

Top-4 Similar Cases:
Case 01   Similarity = 0.94   [Transverse Fatigue Crack - Steel Girder]
Case 18   Similarity = 0.91   [Pitting Corrosion - Offshore Leg]
Case 42   Similarity = 0.87   [Concrete Spalling & Exposed Rebar]
Case 67   Similarity = 0.85   [Weld Toe Micro-Porosity]
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/search` | `POST` | Upload query image; returns Top-K similar cases & severity |
| `/api/cases` | `GET` | List all historical inspection cases |
| `/api/cases/{id}`| `GET` | Retrieve single case details and remediation recommendations |
| `/api/samples` | `GET` | Pre-loaded sample inspection images for 1-click testing |
| `/api/stats` | `GET` | Vector index dimension, case counts, and cluster health |
| `/docs` | `GET` | Interactive Swagger API documentation |
