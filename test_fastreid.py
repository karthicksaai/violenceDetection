import cv2
import numpy as np
import sys
import os

# Ensure we can import from orchestration
sys.path.append(os.getcwd())

from orchestration.fastreid_wrapper import FastReIDExtractor

def test():
    try:
        print("Initializing FastReID...")
        # Use CPU for test to be safe on Mac
        extractor = FastReIDExtractor(device='cpu')
        print("Model Loaded.")
        
        # Dummy Image (H, W, C) - Standard ReID size
        img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        
        print("Extracting...")
        feats = extractor.extract([img])
        
        if len(feats) > 0:
            print(f"Feature Shape: {feats[0].shape}")
            print(f"Norm: {np.linalg.norm(feats[0])}")
            print("✅ FastReID Test Passed!")
        else:
            print("❌ No features returned.")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
