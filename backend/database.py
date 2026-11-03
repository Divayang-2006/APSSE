import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

STATIC_DIR = Path(__file__).resolve().parent / "static"
IMAGE_DIR = STATIC_DIR / "images"
SAMPLE_DIR = STATIC_DIR / "samples"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# 12 Detailed Industrial Damage Cases
CASES_METADATA = [
    {
        "case_id": "Case 01",
        "title": "Transverse Fatigue Crack in High-Tensile Steel Girder",
        "damage_type": "Fatigue Crack",
        "severity_level": "Critical",
        "severity_score": 0.94,
        "structure_type": "Steel I-Beam Girder",
        "inspection_date": "2026-03-14",
        "location": "Bay Bridge Support Span B-12",
        "description": "Propagating transverse crack originating at tension flange weld toe. Length measured at 145mm under dynamic traffic loading.",
        "recommended_action": "Immediate emergency arrest hole drilling and retrofitting with high-strength bolted splice plates."
    },
    {
        "case_id": "Case 18",
        "title": "Severe Pitting Corrosion with Sub-Surface Wall Thinning",
        "damage_type": "Pitting Corrosion",
        "severity_level": "Critical",
        "severity_score": 0.91,
        "structure_type": "Offshore Platform Leg",
        "inspection_date": "2026-02-28",
        "location": "North Sea Jacket Leg Node K-3",
        "description": "Deep localized pitting clusters penetrating beyond 40% of nominal wall thickness in splash zone.",
        "recommended_action": "Ultrasonic thickness mapping, abrasive blasting, and application of structural epoxy composite wrap."
    },
    {
        "case_id": "Case 42",
        "title": "Reinforced Concrete Spalling with Exposed Corroded Rebar",
        "damage_type": "Concrete Spalling",
        "severity_level": "Severe",
        "severity_score": 0.87,
        "structure_type": "Highway Overpass Pier",
        "inspection_date": "2026-01-19",
        "location": "Interstate 80 Overpass Pier 4",
        "description": "De-icing salt infiltration leading to delamination of concrete cover and corrosion expansion of longitudinal reinforcement bars.",
        "recommended_action": "Hydro-demolition of loose concrete, rust passivation on exposed rebar, and repair with micro-silica polymer mortar."
    },
    {
        "case_id": "Case 67",
        "title": "Circumferential Weld Toe Porosity and Lack of Fusion",
        "damage_type": "Weld Defect",
        "severity_level": "Severe",
        "severity_score": 0.85,
        "structure_type": "Pressure Vessel Shell",
        "inspection_date": "2025-11-05",
        "location": "Petrochemical Unit C Vessel V-102",
        "description": "Surface-breaking gas pores and linear lack-of-fusion inclusions along circumferential seam weld cap.",
        "recommended_action": "Phased Array Ultrasonic Testing (PAUT), gouging out weld root defects, and re-welding under GTAW procedure."
    },
    {
        "case_id": "Case 12",
        "title": "High-Velocity Cavitation and Slurry Erosion",
        "damage_type": "Cavitation Erosion",
        "severity_level": "Moderate",
        "severity_score": 0.68,
        "structure_type": "Hydroelectric Turbine Runner",
        "inspection_date": "2025-10-12",
        "location": "Dam Generator Turbine Unit 2",
        "description": "Sponge-like pitted surface erosion across trailing edge of stainless steel runner blades caused by vapor bubble collapse.",
        "recommended_action": "Build-up welding with cavitation-resistant cobalt-base Stellite alloy and precision profile grinding."
    },
    {
        "case_id": "Case 29",
        "title": "Thermal Fatigue Craze Cracking",
        "damage_type": "Thermal Stress Crack",
        "severity_level": "Severe",
        "severity_score": 0.79,
        "structure_type": "Boiler Superheater Header",
        "inspection_date": "2025-09-22",
        "location": "Thermal Power Station Block 3",
        "description": "Network of surface craze cracks caused by severe cyclic temperature gradients during rapid cycling starts.",
        "recommended_action": "Magnetic particle inspection and remaining life assessment calculation; derate operational ramping rate."
    },
    {
        "case_id": "Case 33",
        "title": "Atmospheric Rust Layer and Delamination Flakes",
        "damage_type": "Surface Rust Flaking",
        "severity_level": "Moderate",
        "severity_score": 0.58,
        "structure_type": "Suspension Bridge Cable Anchor",
        "inspection_date": "2025-08-15",
        "location": "South Anchor Vault Chamber",
        "description": "Multilayered stratified iron oxide rust pack forming between steel strand clamp plates.",
        "recommended_action": "Mechanical needle scaling, dehumidification system maintenance, and polyurethane barrier coating application."
    },
    {
        "case_id": "Case 55",
        "title": "Micro-Crack Array in Composite Wind Turbine Blade",
        "damage_type": "Delamination Crack",
        "severity_level": "Moderate",
        "severity_score": 0.62,
        "structure_type": "Wind Turbine Blade Spar",
        "inspection_date": "2025-07-30",
        "location": "Offshore Wind Farm WTG-14",
        "description": "Transverse micro-cracks in glass fiber reinforced polymer resin matrix along maximum deflection zone.",
        "recommended_action": "Vacuum infusion resin injection and installation of aerodynamic load alleviation vortex generators."
    },
    {
        "case_id": "Case 74",
        "title": "Freeze-Thaw Surface Concrete Flaking",
        "damage_type": "Concrete Spalling",
        "severity_level": "Minor",
        "severity_score": 0.38,
        "structure_type": "Railway Tunnel Lining",
        "inspection_date": "2025-06-18",
        "location": "Mountain Pass Tunnel KM 42.8",
        "description": "Shallow surface scaling (5-10mm depth) caused by repeated hydraulic pressure during winter freeze-thaw cycles.",
        "recommended_action": "Apply hydrophobic silane penetrant sealer to prevent further moisture ingress."
    },
    {
        "case_id": "Case 89",
        "title": "Longitudinal Seam Crack in High-Pressure Gas Line",
        "damage_type": "Stress Corrosion Crack",
        "severity_level": "Critical",
        "severity_score": 0.95,
        "structure_type": "Gas Pipeline X70 Steel",
        "inspection_date": "2025-05-11",
        "location": "Pipeline Sector Milepost 312",
        "description": "Branched high-pH stress corrosion crack colony adjacent to fusion line on external pipe surface.",
        "recommended_action": "Depressurize line section, excavate anomaly site, and install full-encirclement steel pressure sleeve."
    },
    {
        "case_id": "Case 92",
        "title": "Superficial Paint Coating Blistering",
        "damage_type": "Coating Degradation",
        "severity_level": "Minor",
        "severity_score": 0.28,
        "structure_type": "Bulk Fuel Storage Tank",
        "inspection_date": "2025-04-02",
        "location": "Terminal Tank TK-404 Roof",
        "description": "Isolated osmotic blisters without substrate steel loss. Primer coat remains intact.",
        "recommended_action": "Spot touch-up scraping and repainting during scheduled turnaround."
    },
    {
        "case_id": "Case 104",
        "title": "Intergranular Stress Corrosion Cracking in Austenitic Piping",
        "damage_type": "Corrosion Cracking",
        "severity_level": "Severe",
        "severity_score": 0.82,
        "structure_type": "Nuclear Secondary Cooling Line",
        "inspection_date": "2025-03-10",
        "location": "Cooling Loop Heat Exchanger Loop 2",
        "description": "Sensitization-induced chromium depletion along grain boundaries leading to micro-fissuring.",
        "recommended_action": "Induction heating stress improvement (IHSI) and weld overlay repair."
    }
]

def generate_procedural_damage_image(case: dict, filename: str, is_query: bool = False) -> str:
    """Generates realistic inspection damage image with visual markers."""
    path = (SAMPLE_DIR if is_query else IMAGE_DIR) / filename
    if path.exists():
        return str(path)

    width, height = 480, 360
    dtype = case.get("damage_type", "Crack")
    score = case.get("severity_score", 0.75)
    
    # Base background texture
    if "Concrete" in case.get("structure_type", ""):
        base_color = (165, 168, 170)
        noise_range = 25
    elif "Steel" in case.get("structure_type", "") or "Weld" in dtype:
        base_color = (130, 135, 142)
        noise_range = 20
    elif "Offshore" in case.get("structure_type", "") or "Rust" in dtype or "Corrosion" in dtype:
        base_color = (150, 120, 100)
        noise_range = 35
    else:
        base_color = (140, 145, 150)
        noise_range = 30

    # Create base noisy texture
    base = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        noise = np.random.randint(-noise_range, noise_range, (height, width))
        channel = np.clip(base_color[c] + noise, 0, 255).astype(np.uint8)
        base[:, :, c] = channel

    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # Draw specific damage patterns
    if "Crack" in dtype:
        # Draw realistic jagged fracture lines
        points = []
        x = width * 0.25
        y = height * 0.3
        points.append((x, y))
        steps = int(25 * (score + 0.5))
        for _ in range(steps):
            x += np.random.uniform(8, 20)
            y += np.random.uniform(-10, 12) + (score * 2)
            points.append((x, y))
        
        # Outer fracture shadow
        for i in range(len(points)-1):
            draw.line([points[i], points[i+1]], fill=(30, 30, 35), width=int(3 + score * 4))
            draw.line([points[i], points[i+1]], fill=(15, 15, 20), width=int(1 + score * 2))
            # Branching micro-cracks
            if np.random.rand() > 0.6:
                bx = points[i][0] + np.random.uniform(-15, 20)
                by = points[i][1] + np.random.uniform(-25, 25)
                draw.line([points[i], (bx, by)], fill=(45, 45, 50), width=2)

    elif "Corrosion" in dtype or "Rust" in dtype:
        # Draw rust and oxidation patches
        for _ in range(int(30 * score)):
            rx = np.random.randint(50, width - 80)
            ry = np.random.randint(50, height - 80)
            rad = np.random.randint(15, int(45 * score + 20))
            rust_colors = [(165, 68, 28), (140, 52, 18), (195, 95, 38), (95, 38, 12)]
            col = rust_colors[np.random.randint(0, len(rust_colors))]
            draw.ellipse([rx - rad, ry - rad, rx + rad, ry + rad], fill=col)

    elif "Spalling" in dtype:
        # Concrete crater and exposed rebar
        cx, cy = width // 2, height // 2
        rw, rh = int(120 * score), int(70 * score)
        draw.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=(85, 88, 92), outline=(50, 50, 55), width=3)
        # Rebar lines
        draw.line([(cx - rw, cy - 10), (cx + rw, cy - 10)], fill=(130, 60, 25), width=6)
        draw.line([(cx - rw, cy + 20), (cx + rw, cy + 20)], fill=(130, 60, 25), width=6)

    elif "Weld" in dtype:
        # Weld seam with porosity
        wx = width // 2
        for y in range(20, height - 20, 10):
            draw.ellipse([wx - 25, y, wx + 25, y + 16], fill=(180, 185, 192), outline=(90, 95, 105), width=2)
        # Pores
        for _ in range(int(20 * score)):
            px = wx + np.random.randint(-18, 18)
            py = np.random.randint(40, height - 40)
            draw.ellipse([px-3, py-3, px+3, py+3], fill=(20, 20, 25))

    # Apply soft Gaussian filter to blend realistically
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Add Industrial Inspection Stamp Overlay
    draw = ImageDraw.Draw(img)
    tag_bg = (20, 25, 35, 200)
    draw.rectangle([10, 10, 260, 75], fill=(20, 25, 35))
    tag_text = f"INSPECTION: {case['case_id']}\nDAMAGE: {dtype}\nSEV-RATING: {score:.2f} [{case['severity_level']}]"
    draw.text((18, 16), tag_text, fill=(240, 245, 255))

    # Target indicator reticle
    reticle_col = (255, 75, 75) if score >= 0.8 else (255, 180, 30)
    draw.rectangle([width - 80, 20, width - 20, 80], outline=reticle_col, width=2)
    draw.line([(width - 50, 25), (width - 50, 75)], fill=reticle_col, width=1)
    draw.line([(width - 75, 50), (width - 25, 50)], fill=reticle_col, width=1)

    img.save(path, quality=92)
    return str(path)

def init_database():
    """Initializes historical damage catalog and generates inspection images."""
    for idx, case in enumerate(CASES_METADATA):
        fname = f"case_{idx+1:02d}.jpg"
        generate_procedural_damage_image(case, fname, is_query=False)
        case["image_url"] = f"/static/images/{fname}"

    # Also generate 3 sample query images for instant testing
    sample_queries = [
        {"case_id": "QUERY_CRACK_01", "damage_type": "Fatigue Crack", "severity_score": 0.93, "severity_level": "Critical", "structure_type": "Steel I-Beam Girder"},
        {"case_id": "QUERY_RUST_02", "damage_type": "Pitting Corrosion", "severity_score": 0.89, "severity_level": "Critical", "structure_type": "Offshore Platform Leg"},
        {"case_id": "QUERY_SPALL_03", "damage_type": "Concrete Spalling", "severity_score": 0.84, "severity_level": "Severe", "structure_type": "Highway Overpass Pier"}
    ]
    for idx, sq in enumerate(sample_queries):
        fname = f"sample_query_{idx+1}.jpg"
        generate_procedural_damage_image(sq, fname, is_query=True)

# Run initialization on import
init_database()


import json
DATASET_JSON = Path(__file__).resolve().parent / "dataset_cases.json"

def get_combined_catalog():
    """Returns combined catalog of industrial base cases and ingested concrete crack cases."""
    cases = list(CASES_METADATA)
    if DATASET_JSON.exists():
        try:
            with open(DATASET_JSON, "r", encoding="utf-8") as f:
                extra_cases = json.load(f)
                cases.extend(extra_cases)
        except Exception:
            pass
    return cases
