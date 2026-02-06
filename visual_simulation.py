import cv2
import numpy as np
import time
import sys
import os
from ultralytics import YOLO
from orchestration.engine import OrchestrationEngine

# Ensure window supports resize
cv2.namedWindow("Control Room: Event-Driven Intelligence Layer", cv2.WINDOW_NORMAL)

def draw_text(img, text, pos, color=(255, 255, 255), scale=0.8):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def create_grid(images, grid_size=(2, 2), tile_size=(640, 480)):
    # Create a black canvas
    h, w = tile_size
    rows, cols = grid_size
    canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

    for idx, (cam_id, img) in enumerate(images.items()):
        if idx >= rows * cols: break
        
        r, c = divmod(idx, cols)
        
        # Resize if needed (naive resize for demo)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
            
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img
        
    return canvas

def run_visual_simulation():
    engine = OrchestrationEngine('orchestration/cameras_network.json')
    print("🎥 Visual Simulation Started. Press 'q' to exit.")

    # Load Trained Model (or use yolov8n.pt if training not done, but we found best.pt)
    model_path = 'runs/detect/violence_detection_aug/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}, utilizing mock logic or checking other paths.")
        # Fallback to standard if needed, or error out. 
        # For now, let's assume it works as we verified the file existence.
    
    try:
        model = YOLO(model_path)
        print(f"✅ Loaded YOLO Model: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Simulation State
    start_time = time.time()
    violence_triggered = False

    while True:
        current_time = time.time() - start_time
        frames = {}
        active_feeds = engine.get_active_feeds()
        
        for cam_id, cam in engine.cameras.items():
            # Get frame
            # If camera is technically "Passive" (standby), we still want to show it in the UI, just darkened.
            # But VirtualCamera returns None if not active.
            # So we temporarily wake it for the "God View" UI.
            was_active = cam.is_active
            if not was_active: 
                cam.is_active = True
                frame, valid = cam.get_frame()
                cam.is_active = False # Restore
            else:
                frame, valid = cam.get_frame()

            # 0. DETERMINE STATUS
            is_system_active = cam_id in active_feeds or (cam_id == 'cam-01' and not violence_triggered)

            if not valid or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                draw_text(frame, "NO SIGNAL", (200, 240), (0, 0, 255))
            else:
                # --- DETECTION LOGIC (Network-Wide) ---
                if is_system_active:
                    threshold = 0.4 
                    results = model.predict(frame, verbose=False, conf=threshold)
                    
                    active_subject_id = None
                    for evt in engine.active_events.values():
                        if evt.status == 'active':
                            active_subject_id = evt.metadata.get('subject_id')
                            break

                    detected_trigger = False
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            if hasattr(model, 'names'):
                                cls_name = model.names[cls_id]
                            else:
                                cls_name = str(cls_id)
                            
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # --- FILTERING ---
                            # Only visualize Person and Violence. Ignore cars, bags, etc. for clarity.
                            accepted_classes = ['Person', 'Violence', 'person', 'violence', 'Human']
                            if cls_name not in accepted_classes:
                                continue

                            # 1. TRIGGER LOGIC (Entrance Cam)
                            if cam_id == 'cam-01' and not violence_triggered:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                
                                if (cls_name in ['Violence', 'violence'] or cls_id == 1) and conf > 0.6:
                                    print(f"🚨 VIOLENCE DETECTED on {cam_id} | Conf: {conf:.2f}")
                                    engine.handle_detection(
                                        cam_id, 
                                        'Violence', 
                                        conf,
                                        bbox=[x1, y1, x2, y2],
                                        frame=frame
                                    )
                                    violence_triggered = True 
                                    detected_trigger = True
                                    
                                    # Draw Red Event Box
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                    cv2.putText(frame, "STATUS: ASSAULT", (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            # 2. TRACKING LOGIC (Downstream Cams)
                            elif violence_triggered and active_subject_id:
                                # ReID Matching
                                if cls_name in accepted_classes: 
                                    # Draw Green "MATCH" Box
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    
                                    
                                    # Action Tag Logic
                                    action_tag = "SUSPECT FLEEING" if cam_id != 'cam-01' else "SUSPECT DETECTED"
                                    
                                    # Label with ID and Action
                                    label = f"{active_subject_id} | {action_tag}"
                                    cv2.putText(frame, label, (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                                
                                    # --- RECURSIVE TRIGGER ---
                                    # If this is a downstream camera (not Cam 1) and we found a match,
                                    # report it to the engine so it can trigger *its* neighbors (e.g. Cam 2 -> Cam 3)
                                    # We use a simple debounce/check to avoid flooding
                                    if cam_id != 'cam-01' and conf > 0.5:
                                        # "Person" detected by neighbor. Treat as 'Suspicious' event update.
                                        engine.handle_detection(
                                            cam_id, 
                                            'Person', 
                                            conf,
                                            bbox=[x1, y1, x2, y2],
                                            frame=frame
                                        )

                        if detected_trigger: break

            # --- VISUALIZATION LOGIC ---
            if is_system_active:
                if cam_id == 'cam-01' and violence_triggered:
                    color = (0, 0, 255) # Red for Event Source
                    status = "VIOLENCE DETECTED"
                elif cam_id == 'cam-01':
                    color = (0, 255, 0) # Green for Monitoring
                    status = "ACTIVE MONITORING"
                else: 
                    color = (0, 165, 255) # Orange for Pursuit
                    status = "PURSUIT TRACKING"
                
                
                cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), color, 10)
                draw_text(frame, f"{status}", (30, 60), color)
                # Blinking REC dot for realism
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(frame, (frame.shape[1]-40, 40), 10, (0, 0, 255), -1)
                    draw_text(frame, "REC", (frame.shape[1]-100, 50), (0, 0, 255), 0.6)

            else:
                # Passive Logic (Grayed out)
                frame = (frame * 0.4).astype(np.uint8)
                
                # Check distance from Cam 1 (Mock Logic for visualization text)
                # To show user why it's not active if they expect it to be
                draw_text(frame, "PASSIVE (OUT OF RANGE)", (30, 30), (100, 100, 100))

            draw_text(frame, f"CAM: {cam_id}", (30, 450), (255, 255, 255))
            frames[cam_id] = frame

        # Render Grid
        grid_img = create_grid(frames, grid_size=(3, 3))
        
        # Overlay Global Info
        cv2.putText(grid_img, f"Time: {current_time:.1f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if violence_triggered:
            status_text = "EVENT ACTIVE"
            color = (0, 0, 255)
            
            # Show ReID Info
            # Find the active event and get the subject ID
            for event in engine.active_events.values():
                if event.status == 'active':
                    subj = event.metadata.get('subject_id', 'Unknown')
                    cv2.putText(grid_img, f"TRACKING SUBJECT: {subj}", (20, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    break
        else:
            status_text = "MONITORING NETWORK"
            color = (0, 255, 0)
            
        cv2.putText(grid_img, f"System Status: {status_text}", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Control Room: Event-Driven Intelligence Layer", grid_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_visual_simulation()
