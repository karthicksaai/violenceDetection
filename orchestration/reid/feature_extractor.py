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
        print(f"🧠 Initializing ReID Feature Extractor (OSNet) on {self.device}...")
        
        # Load OSNet (Pretrained)
        try:
            from orchestration.reid.osnet import osnet_x1_0
            # pretrained=True will look in ~/.cache/torch/checkpoints/ found by gdown
            self.model = osnet_x1_0(pretrained=True)
        except ImportError:
            print("❌ Error importing OSNet. Please ensure orchestration.reid.osnet exists.")
            raise
        except Exception as e:
            print(f"❌ Error loading OSNet weights: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()
        
        # OSNet Preprocessing (Standard ImageNet)
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        print("✅ ReID Model (OSNet) Loaded Successfully.")

    def extract(self, img_numpy):
        """
        Extract feature vector from an OpenCV image (BGR numpy array).
        Returns: numpy array of shape (512,)
        """
        try:
            # Convert BGR (OpenCV) to RGB
            img_rgb = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB) if 'cv2' in globals() else img_numpy[:, :, ::-1]
            
            # Convert to PIL for transform
            img_pil = Image.fromarray(img_rgb)
            
            # Preprocess
            input_tensor = self.preprocess(img_pil)
            
            # TTA: Horizontal Flip
            input_tensor_flip = self.preprocess(img_pil.transpose(Image.FLIP_LEFT_RIGHT))
            
            # Batch of 2 (Original + Flip)
            input_batch = torch.stack([input_tensor, input_tensor_flip]).to(self.device)
            
            # Inference
            with torch.no_grad():
                features = self.model(input_batch) # Returns (2, 512)
                
            # Average Features (TTA)
            # features[0] is original, features[1] is flipped
            embedding = features[0] + features[1] 
            
            # Flatten & Normalize
            embedding = torch.flatten(embedding).cpu().numpy()
            
            # L2 Normalize
            norm = np.linalg.norm(embedding)
            return embedding / (norm + 1e-6)
            
        except Exception as e:
            print(f"❌ Feature Extraction Failed: {e}")
            return np.zeros(512)

    @staticmethod
    def cosine_similarity(emb1, emb2):
        """
        Compute Cosine Similarity between two L2-normalized vectors.
        Range: -1 to 1. (Higher is better match)
        """
        if emb1 is None or emb2 is None: return 0.0
        return np.dot(emb1, emb2)
