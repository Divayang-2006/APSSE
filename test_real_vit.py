import sys, json, numpy as np
from PIL import Image
sys.path.insert(0, r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\backend")
import models

models._FORCE_FALLBACK = False
models.TORCH_AVAILABLE = True
ext = models.FeatureExtractor("vit")
print("Model is real ViT:", ext.model is not None)

emb = np.load(r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\backend\embeddings_vit.npy")
with open(r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\backend\dataset_cases.json", encoding="utf-8") as f:
    cat = json.load(f)

img1 = Image.open(r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\dataset\extracted\Positive\00001.jpg").convert("RGB")
q1 = ext.extract(img1)
scores1 = emb @ q1
top_idx1 = np.argsort(scores1)[::-1][:5]
print("\n--- Query: Positive/00001.jpg ---")
for idx in top_idx1:
    cid = cat[idx]["case_id"]
    dtype = cat[idx]["damage_type"]
    print(f"{cid} ({dtype}): Similarity = {scores1[idx]:.4f}")

img2 = Image.open(r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\dataset\extracted\Positive\00002.jpg").convert("RGB")
q2 = ext.extract(img2)
scores2 = emb @ q2
top_idx2 = np.argsort(scores2)[::-1][:5]
print("\n--- Query: Positive/00002.jpg ---")
for idx in top_idx2:
    cid = cat[idx]["case_id"]
    dtype = cat[idx]["damage_type"]
    print(f"{cid} ({dtype}): Similarity = {scores2[idx]:.4f}")

