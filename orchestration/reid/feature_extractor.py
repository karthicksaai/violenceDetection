import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"🧠 Initializing ReID Feature Extractor (ResNet50) on {self.device}...")
        
        # Load Pretrained ResNet50
        # We strip the final classification layer to get the feature embedding
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        
        # Remove the fully connected layer (fc) to get raw embeddings (2048 dim)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 128)), # Standard ReID size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        print("✅ ReID Model Loaded Successfully.")

    def extract(self, img_numpy):
        """
        Extract feature vector from an OpenCV image (BGR numpy array).
        Returns: numpy array of shape (2048,)
        """
        try:
            # Convert BGR (OpenCV) to RGB
            img_rgb = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB) if 'cv2' in globals() else img_numpy[:, :, ::-1]
            
            # Convert to PIL for transform
            img_pil = Image.fromarray(img_rgb)
            
            # Preprocess
            input_tensor = self.preprocess(img_pil)
            input_batch = input_tensor.unsqueeze(0).to(self.device) # Batch of 1
            
            # Inference
            with torch.no_grad():
                embedding = self.model(input_batch)
                
            # Flatten: [1, 2048, 1, 1] -> [2048]
            embedding = torch.flatten(embedding).cpu().numpy()
            
            # L2 Normalize (Essential for Cosine Similarity)
            norm = np.linalg.norm(embedding)
            return embedding / norm
            
        except Exception as e:
            print(f"❌ Feature Extraction Failed: {e}")
            return np.zeros(2048)

    @staticmethod
    def cosine_similarity(emb1, emb2):
        """
        Compute Cosine Similarity between two L2-normalized vectors.
        Range: -1 to 1. (Higher is better match)
        """
        if emb1 is None or emb2 is None: return 0.0
        return np.dot(emb1, emb2)
