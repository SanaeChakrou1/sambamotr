import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

class ReIDExtractor(nn.Module):
    def __init__(self, model_path=None, device='cuda'):
        super(ReIDExtractor, self).__init__()
        # Utilisation de ResNet18 comme extracteur de caractéristiques de base
        # Dans un scénario réel, on utiliserait un modèle entraîné sur Market-1501
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Supprimer la couche de classification pour obtenir les embeddings
        self.model.to(device)
        self.model.eval()
        
        self.device = device
        self.preprocess = T.Compose([
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image, bboxes):
        """
        image: Tensor (C, H, W)
        bboxes: Tensor (N, 4) en format [x1, y1, x2, y2] (normalisé ou non)
        """
        if len(bboxes) == 0:
            return torch.empty((0, 512)).to(self.device)
            
        crops = []
        h, w = image.shape[-2:]
        
        # S'assurer que l'image est sur CPU pour le découpage si nécessaire, 
        # ou utiliser des opérations de grille si sur GPU.
        # Pour la simplicité, on utilise PIL ou des découpes directes.
        
        for box in bboxes:
            x1, y1, x2, y2 = box
            # Conversion en pixels si normalisé
            if x1 < 1.0 and y1 < 1.0 and x2 <= 1.0 and y2 <= 1.0:
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
            
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                crops.append(torch.zeros((3, 128, 64)).to(self.device))
                continue
                
            crop = image[:, y1:y2, x1:x2]
            # Redimensionnement et normalisation
            crop_pil = T.ToPILImage()(crop.cpu())
            crops.append(self.preprocess(crop_pil).to(self.device))
            
        if not crops:
            return torch.empty((0, 512)).to(self.device)
            
        batch = torch.stack(crops)
        features = self.model(batch)
        # Normalisation L2 pour la similarité cosinus
        features = F.normalize(features, p=2, dim=1)
        return features
