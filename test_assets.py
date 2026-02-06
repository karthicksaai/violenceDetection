import cv2
import os
from ultralytics import YOLO

def test_assets():
    print("--- DIAGNOSTIC START ---")
    
    # 1. Check Video Files
    video_files = [
        "/Users/yuva/development/violenceDetection/CameraTest/Camera1.mp4",
        "/Users/yuva/development/violenceDetection/CameraTest/Camera2.mp4",
        "/Users/yuva/development/violenceDetection/CameraTest/Camera3.mp4"
    ]
    
    for v in video_files:
        if not os.path.exists(v):
            print(f"❌ File missing: {v}")
        else:
            cap = cv2.VideoCapture(v)
            if not cap.isOpened():
                print(f"❌ Failed to open video (Codec issue?): {v}")
            else:
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Video works: {os.path.basename(v)} | Res: {frame.shape}")
                else:
                    print(f"⚠️ Video opened but failed to read frame: {v}")
                cap.release()

    # 2. Check Model
    model_path = 'runs/detect/violence_detection_aug/weights/best.pt'
    print(f"\nChecking Model: {model_path}")
    if os.path.exists(model_path):
        try:
            model = YOLO(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
            # Try dummy inference
            # model.predict(source=video_files[0], conf=0.5, max_det=1)
        except Exception as e:
            print(f"❌ Model load failed: {e}")
    else:
        print("❌ Model file does not exist")

    print("\n--- DIAGNOSTIC END ---")

if __name__ == "__main__":
    test_assets()
