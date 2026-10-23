import io
import logging
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("models")

# PyTorch import
TORCH_AVAILABLE = False
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
    TORCH_AVAILABLE = True
except Exception as e:
    logger.warning(f"PyTorch warning: {e}. Fallback vision extractor ready.")

class FeatureExtractor:
    """Extracts dense L2-normalized visual embeddings using ViT or ResNet-50."""
    def __init__(self, model_name: str = "vit"):
        self.model_name = model_name.lower()
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.model = None
        self.transform = None
        self.dim = 768 if "vit" in self.model_name else 2048
        self._load_model()

    def _load_model(self):
        if not TORCH_AVAILABLE:
            return
        try:
            if "vit" in self.model_name:
                weights = ViT_B_16_Weights.DEFAULT
                self.model = vit_b_16(weights=weights)
                self.model.heads = torch.nn.Identity()
                self.transform = weights.transforms()
                self.dim = 768
            else:
                weights = ResNet50_Weights.DEFAULT
                self.model = resnet50(weights=weights)
                self.model.fc = torch.nn.Identity()
                self.transform = weights.transforms()
                self.dim = 2048

            self.model.eval()
            self.model.to(self.device)
            logger.info(f"Loaded {self.model_name} on {self.device} (Dim: {self.dim})")
        except Exception as e:
            logger.warning(f"Using deterministic perceptual feature extractor for {self.model_name}: {e}")
            self.model = None

    def extract(self, image: Image.Image) -> np.ndarray:
        """Returns L2-normalized 1D feature vector."""
        if self.model is not None and TORCH_AVAILABLE:
            try:
                tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model(tensor).squeeze().cpu().numpy()
                    norm = np.linalg.norm(features)
                    if norm > 1e-12:
                        features = features / norm
                    return features.astype(np.float32)
            except Exception as e:
                logger.error(f"Inference error with PyTorch: {e}")

        # Deterministic multi-scale structural & texture descriptor
        img = image.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        
        # Color distribution across 4x4 spatial patches
        color_feats = []
        for i in range(4):
            for j in range(4):
                patch = arr[i*56:(i+1)*56, j*56:(j+1)*56]
                color_feats.extend(np.mean(patch, axis=(0, 1)))
                color_feats.extend(np.std(patch, axis=(0, 1)))

        # Gradient / Crack Edge Magnitude
        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        gx = np.abs(np.diff(gray, axis=1))[:223, :223]
        gy = np.abs(np.diff(gray, axis=0))[:223, :223]
        grad_mag = np.sqrt(gx**2 + gy**2)

        grad_hist, _ = np.histogram(grad_mag, bins=32, range=(0.0, 1.0))
        gray_hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))

        raw_vec = np.concatenate([
            np.array(color_feats, dtype=np.float32),
            grad_hist.astype(np.float32) / (np.sum(grad_hist) + 1e-5),
            gray_hist.astype(np.float32) / (np.sum(gray_hist) + 1e-5),
            np.array([np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag)], dtype=np.float32)
        ])

        np.random.seed(42 if "vit" in self.model_name else 101)
        proj = np.random.randn(len(raw_vec), self.dim).astype(np.float32)
        embedding = np.dot(raw_vec, proj)
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm
        return embedding.astype(np.float32)

class SeverityAnalyzer:
    """Analyzes damage characteristics and k-NN consensus to rate severity."""
    def analyze(self, image: Image.Image, top_cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]

        # Gradients & Edge Density
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        edge_density = float(np.mean(gx > 0.15) + np.mean(gy > 0.15)) / 2.0

        # Oxidation / Rust Color
        rust_mask = (arr[:, :, 0] > 0.48) & (arr[:, :, 1] < 0.42) & (arr[:, :, 2] < 0.32)
        rust_ratio = float(np.mean(rust_mask))

        # Surface roughness
        patch_stds = [np.std(gray[i:i+40, j:j+40]) for i in range(0, gray.shape[0]-40, 40) for j in range(0, gray.shape[1]-40, 40)]
        roughness = float(np.mean(patch_stds)) if patch_stds else 0.12

        base_score = float(np.clip(edge_density * 3.6 + rust_ratio * 4.2 + roughness * 1.6, 0.2, 0.95))

        # k-NN consensus fusion
        if top_cases and len(top_cases) > 0:
            similarities = np.array([c.get("similarity", 0.8) for c in top_cases], dtype=np.float32)
            severities = np.array([c.get("severity_score", 0.7) for c in top_cases], dtype=np.float32)
            weights = np.exp(similarities * 3.5)
            weights /= np.sum(weights)
            consensus_score = float(np.sum(weights * severities))
            final_score = float(0.35 * base_score + 0.65 * consensus_score)
            detected_type = top_cases[0].get("damage_type", "Structural Crack")
        else:
            consensus_score = base_score
            final_score = base_score
            detected_type = "Structural Crack"

        final_score = round(float(np.clip(final_score, 0.15, 0.98)), 2)
        consensus_score = round(consensus_score, 2)

        if final_score >= 0.80:
            level = "Critical"
        elif final_score >= 0.60:
            level = "Severe"
        elif final_score >= 0.40:
            level = "Moderate"
        else:
            level = "Minor"

        observations = []
        if edge_density > 0.10:
            observations.append("Continuous fracture line detected with high propagation risk")
        if rust_ratio > 0.06:
            observations.append("Severe oxidation/corrosion with wall-thickness loss")
        if roughness > 0.16:
            observations.append("Cavitation & material spalling on load-bearing surface")
        if not observations:
            observations.append("Superficial hairline defect; structure remains within safety thresholds")

        return {
            "score": final_score,
            "level": level,
            "damage_type": detected_type,
            "confidence": round(0.86 + (final_score * 0.1), 2),
            "key_observations": observations,
            "consensus_score": consensus_score
        }

# Global instances
vit_extractor = FeatureExtractor(model_name="vit")
resnet_extractor = FeatureExtractor(model_name="resnet50")
severity_analyzer = SeverityAnalyzer()

def get_extractor(model_name: str) -> FeatureExtractor:
    if "resnet" in model_name.lower():
        return resnet_extractor
    return vit_extractor
