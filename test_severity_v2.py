import numpy as np
from PIL import Image

def analyze(img):
    arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    ed_fine = float((np.mean(gx > 0.04) + np.mean(gy > 0.04)) / 2.0)
    ed_strong = float((np.mean(gx > 0.08) + np.mean(gy > 0.08)) / 2.0)
    mean_val = float(np.mean(gray))
    dark_ratio = float(np.mean(gray < (mean_val - 0.15)))
    grad_mag = np.sqrt(gx[:gy.shape[0], :]**2 + gy[:, :gx.shape[1]]**2)
    p95 = float(np.percentile(grad_mag, 95))

    raw = (ed_fine * 6.0) + (ed_strong * 25.0) + (dark_ratio * 8.0) + (p95 * 1.5)
    base_score = float(np.clip(raw, 0.05, 0.98))

    if base_score >= 0.80:
        level = 'Critical'
        finding = 'Extensive open crack network detected with structural propagation risk.'
        dtype = 'Concrete Structural Crack'
    elif base_score >= 0.60:
        level = 'Severe'
        finding = 'Significant linear fracture line with edge spalling and depth penetration.'
        dtype = 'Concrete Structural Crack'
    elif base_score >= 0.35:
        level = 'Moderate'
        finding = 'Visible surface crack detected; monitoring and surface sealing recommended.'
        dtype = 'Concrete Structural Crack'
    else:
        level = 'Minor'
        finding = 'Monolithic intact concrete surface; no fracture lines or spalling detected.'
        dtype = 'Intact Concrete Surface'

    return {'score': round(base_score, 2), 'level': level, 'damage_type': dtype, 'finding': finding}

for p in ['dataset/extracted/Positive/00001.jpg', 'dataset/extracted/Positive/00010.jpg', 'dataset/extracted/Positive/00100.jpg', 'dataset/extracted/Negative/00001.jpg', 'dataset/extracted/Negative/00020.jpg']:
    res = analyze(Image.open(p))
    sc = res['score']
    lvl = res['level']
    dt = res['damage_type']
    fnd = res['finding']
    print(f"{p[-20:]} => {sc} | {lvl} | {dt} | {fnd[:45]}...")

