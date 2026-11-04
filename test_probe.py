import requests
with open(r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\dataset\extracted\Positive\00001.jpg", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/api/search", files={"file": ("00001.jpg", f, "image/jpeg")}, data={"model": "vit", "top_k": 4})
data = r.json()
print("Status:", data.get("status"))
print("Model used:", data.get("model_used"))
print("Vector DB:", data.get("vector_database"))
print("Results:")
for hit in data.get("results", []):
    cid = hit.get("case_id")
    sim = hit.get("similarity")
    dtype = hit.get("damage_type")
    print(f"  {cid} ({dtype}) => Similarity = {sim}")

