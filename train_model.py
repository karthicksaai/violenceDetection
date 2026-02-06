from ultralytics import YOLO
import os

def train_violence_model():
    # Define dataset path
    dataset_yaml = '/Users/yuva/development/violenceDetection/Violence.v1i.yolov8/data.yaml'
    
    # Load a model
    # We use 'yolov8n.pt' (nano) for speed. 
    # For better accuracy in production, we would use 'yolov8m.pt' or 'yolov8l.pt'.
    model = YOLO('yolov8n.pt')  

    # Train the model
    print("Starting training...")
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        name='violence_detection_aug',
        project='/Users/yuva/development/violenceDetection/runs/detect',
        # Data Augmentation for Robustness (Fog, Blur, Lighting)
        hsv_h=0.015,  # Hue adjustments
        hsv_s=0.7,    # Saturation variance (simulates color shift)
        hsv_v=0.4,    # Value variance (simulates lighting/darkness)
        degrees=10.0, # Rotation
        translate=0.1,
        scale=0.5,    # Scaling
        mosaic=1.0,   # Strong mosaic augmentation
        fliplr=0.5,
    )
    
    print("Training complete. Model saved to runs/detect/violence_detection_v1/weights/best.pt")
    
    # Validate the model
    metrics = model.val()
    print(f"Validation map50-95: {metrics.box.map}")

if __name__ == '__main__':
    train_violence_model()
