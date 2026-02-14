import cv2
import sys
import os
import torch
from orchestration.yolo_orchestrator import YOLOOrchestrator

# Add project root to path
sys.path.append(os.getcwd())

def test_pipeline(video_path):
    print(f"🎬 Testing YOLOv26 Pipeline on {video_path}")
    
    # Initialize Orchestrator
    try:
        # Use a dummy model path if the user hasn't provided the real one yet, 
        # but the code expects "yolov26.pt". 
        # If it fails, we catch it.
        orchestrator = YOLOOrchestrator(yolo_model="yolo26n.pt")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        print("ℹ️  Ensure 'yolov26.pt' is in the root directory.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Process
        results = orchestrator.process_frame(frame)
        
        # Output
        if results:
            v_score = results.get('violence_score', 0.0)
            c_score = results.get('crowd_score', 0.0)
            alert = results.get('alert')
            
            status = f"Frame {frame_count}: V={v_score:.2f} | C={c_score:.2f}"
            if alert:
                status += f" | 🚨 {alert} -> {results.get('action')}"
            
            print(status)
        else:
            print(f"Frame {frame_count}: No Analysis (Buffering or No Detections)")
            
        frame_count += 1
        if frame_count > 100: break # Test first 100 frames

    cap.release()
    print("✅ Test Complete")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_yolo_pipeline.py <video_path>")
    else:
        test_pipeline(sys.argv[1])
