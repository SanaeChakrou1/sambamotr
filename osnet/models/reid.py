import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchreid.utils import FeatureExtractor

class ReIDExtractor(nn.Module):
    def __init__(self, model_name='osnet_x1_0', model_path=None, device='cuda'):
        super(ReIDExtractor, self).__init__()
        # Utilisation de FeatureExtractor de torchreid pour OsNet
        # Si model_path est None, il téléchargera les poids pré-entraînés par défaut
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device
        )
        self.device = device

    @torch.no_grad()
    def extract(self, image, bboxes):
        """
        image: Tensor (C, H, W) ou (1, C, H, W)
        bboxes: Tensor (N, 4) en format [x1, y1, x2, y2] (normalisé ou non)
        """
        if len(bboxes) == 0:
            return torch.empty((0, 512)).to(self.device)
            
        # S'assurer que l'image est un tenseur (C, H, W)
        if image.dim() == 4:
            image = image[0]
            
        h, w = image.shape[-2:]
        crops = []
        
        # L'extracteur de torchreid attend des images PIL ou des numpy arrays (BGR)
        # On va préparer des numpy arrays pour chaque crop
        image_np = image.permute(1, 2, 0).cpu().numpy()
        # Conversion RGB -> BGR pour torchreid si nécessaire (FeatureExtractor gère souvent le RGB/BGR en interne selon l'entrée)
        # Mais par défaut, il attend du BGR si c'est du numpy.
        image_np = (image_np * 255).astype(np.uint8)
        image_bgr = image_np[:, :, ::-1] # RGB to BGR
        
        for box in bboxes:
            x1, y1, x2, y2 = box
            # Conversion en pixels si normalisé
            if x1 < 1.0 and y1 < 1.0 and x2 <= 1.0 and y2 <= 1.0:
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
            
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                # Crop vide ou invalide, on met un placeholder (noir)
                crops.append(np.zeros((128, 64, 3), dtype=np.uint8))
                continue
                
            crop = image_bgr[y1:y2, x1:x2]
            crops.append(crop)
            
        if not crops:
            return torch.empty((0, 512)).to(self.device)
            
        # FeatureExtractor gère le batching et le preprocessing (resize, normalize)
        features = self.extractor(crops)
        
        # features est un tenseur torch retourné par FeatureExtractor
        # On s'assure qu'il est sur le bon device et normalisé L2
        features = features.to(self.device)
        features = F.normalize(features, p=2, dim=1)
        
        return features
