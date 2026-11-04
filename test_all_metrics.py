import requests

test_images = [
    ("Positive/00001.jpg", r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\dataset\extracted\Positive\00001.jpg"),
    ("Positive/00010.jpg", r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\dataset\extracted\Positive\00010.jpg"),
    ("Negative/00001.jpg", r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\dataset\extracted\Negative\00001.jpg"),
    ("Negative/00050.jpg", r"c:\Users\Saahil Saitwal\OneDrive\Desktop\DESIGN_PROJECT\APSSE\dataset\extracted\Negative\00050.jpg")
]

for name, path in test_images:
    with open(path, "rb") as f:
        r = requests.post("http://127.0.0.1:8000/api/search", files={"file": (name, f, "image/jpeg")}, data={"model": "vit", "top_k": 4})
    data = r.json()
    sev = data["severity_analysis"]
    print("=" * 60)
    print(f"QUERY: {name}")
    print(f"  Classification: {sev['damage_type']}")
    print(f"  Severity Level: {sev['level']}")
    print(f"  Severity Score: {sev['score']} ({int(sev['score']*100)}%)")
    print(f"  Consensus Score: {sev['consensus_score']}")
    print(f"  Key Finding: {sev['key_observations'][0]}")
    print("  Top Matches:")
    for hit in data["results"][:3]:
        print(f"    {hit['case_id']} | Sim={hit['similarity']} | Sev={hit['severity_score']} ({hit['severity_level']}) | {hit['damage_type']}")

