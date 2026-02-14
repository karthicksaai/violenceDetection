import cv2
import numpy as np
import time
import sys
import os
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

# --- GLOBAL STATE FOR UI INTERACTION ---
click_registry = {} # {cam_id: [(bbox, track_id), ...]}
cam_order = []      # [cam_id_at_idx_0, cam_id_at_idx_1, ...]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        engine = param['engine']
        
        # Grid Config (Matches create_grid)
        H_HEADER = 80
        W_TILE = 640
        H_TILE = 480
        COLS = 3
        
        if y < H_HEADER: return
        
        gy = y - H_HEADER
        gx = x
        
        col = gx // W_TILE
        row = gy // H_TILE
        idx = row * COLS + col
        
        # Identify Camera
        if idx < len(cam_order):
            cam_id = cam_order[idx]
            local_x = gx % W_TILE
            local_y = gy % H_TILE
            
            # Check against Registered Suspect Boxes
            if cam_id in click_registry:
                for bbox, track_id in click_registry[cam_id]:
                    bx1, by1, bx2, by2 = bbox
                    if bx1 <= local_x <= bx2 and by1 <= local_y <= by2:
                         print(f"🖱️ UI FEEDBACK: Marking Track {track_id} as FALSE POSITIVE on {cam_id}")
                         engine.handle_feedback(cam_id, track_id, 'false_positive')
                         
                         # Visual Feedback (Short term)
                         # We can't draw here easily without threading issues, 
                         # but the engine state update will clear it next frame.
                         return

def run_visual_simulation():
    # Helper to check for model before crashing
    if not os.path.exists("yolo26n.pt"):
        print("❌ Error: 'yolo26n.pt' not found in root directory.")
        return

    try:
        engine = OrchestrationEngine('orchestration/cameras_network.json')
    except Exception as e:
        print(f"❌ Failed to initialize Engine: {e}")
        return

    print("🎥 Visual Simulation Started. Press 'q' to exit.")
    
    # Setup Window & Callback
    window_name = "Control Room: Event-Driven Intelligence Layer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, {'engine': engine})

    # Simulation State
    start_time = time.time()
    violence_triggered = False
    
    # Store Dispatcher Reference
    dispatcher = engine.dispatcher

    while True:
        current_time = time.time() - start_time
        frames = {}
        active_feeds = engine.get_active_feeds()
        
        # Reset Registry & Order for this frame
        global click_registry, cam_order
        click_registry = {}
        cam_order = []
        
        # Consistent Iteration for Grid Stability
        sim_cameras = sorted(list(engine.cameras.items()), key=lambda x: x[0])
        
        for cam_id, cam in sim_cameras:
            cam_order.append(cam_id)
            
            # Get frame (Simulating active feed for Cam-01 even if passive in engine)
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
                # --- INTELLIGENCE LAYER ---
                gru_conf = 0.0
                
                if is_system_active:
                    # 1. ORCHESTRATOR INFERENCE
                    # We process frame through the engine's dispatcher
                    # Note: In real engine, this is done in process_camera_stream
                    
                    # Optimization: Get confirmed tracks to skip redundant extraction
                    confirmed_ids = set()
                    for ids in engine.confirmed_suspects.values():
                         confirmed_ids.update(ids)  

                    # Run Analysis from Dispatcher (YOLO + OSNet + GRU)
                    analysis = dispatcher.process_frame(
                        frame,
                        camera_id=cam_id
                    )

                    
                    # PROCESS Orchestration Logic (This triggers ReID & Alerts)
                    matched_event_id = engine.process_analysis(cam_id, analysis, frame)
                    
                    gru_conf = 0.0
                    if analysis:
                        v_score = analysis.get('violence_score', 0.0)
                        c_score = analysis.get('crowd_score', 0.0)
                        alert_type = analysis.get('alert')
                        gru_conf = v_score if v_score > c_score else c_score
                        
                        if alert_type and not violence_triggered and cam_id == 'cam-01':
                            print(f"🚨 {alert_type} DETECTED on {cam_id} | Conf: {gru_conf:.2f}")
                            violence_triggered = True

                    # 3. VISUALIZATION OVERLAY
                    # We re-run YOLO predict just to get boxes for drawing (since API consumes them internally)
                    # Ideally orchestrator returns boxes, but for now we do this for UI
                    results = dispatcher.yolo(frame, verbose=False)
                    detected_trigger = False
                    
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            # Handle names safely
                            if hasattr(dispatcher.yolo, 'names'):
                                label = dispatcher.yolo.names[cls]
                            else:
                                label = str(cls)
                            
                            color = (0, 255, 0)
                            label_text = f"{label} {conf:.2f}"
                            
                            # Highlight Violence Class specifically if present
                            if label.lower() in ['violence', 'fight']:
                                color = (0, 0, 255)
                                label_text = f"VIOLENCE {conf:.2f}"
                            
                            # Highlight Suspect (ReID Match context) - ONLY IF MATCHED
                            # Note: In real system we match bbox to analysis['bbox']. 
                            # Here we assume the largest person is the subject if matched_event_id is present.
                            # Better: Check against analysis['bbox'] if available.
                            # Highlight Suspect (ReID Match context) - ONLY IF MATCHED
                            elif label.lower() == 'person':
                                is_suspect = False
                                tracked_id = int(box.id[0].item()) if box.id is not None else -1
                                match_info = analysis.get('matched_details')
                                
                                if cam_id == 'cam-01':
                                    if not violence_triggered:
                                         # Reset Origin Tracking if violence stops
                                         if 'origin_embedding' in globals():
                                             del globals()['origin_embedding']
                                         # Reset Confirmation Counters
                                         if 'suspect_counters' in globals():
                                             globals()['suspect_counters'].clear()
                                    
                                    elif violence_triggered:
                                        # Init Counters if needed
                                        if 'suspect_counters' not in globals():
                                            global suspect_counters
                                            suspect_counters = {}

                                        # Origin Camera: Track by Appearance (Stable)
                                        # 1. Init Reference Feature if needed
                                        if 'origin_embedding' not in globals() and analysis.get('detections'):
                                             # Assume primary suspect is the first detection at trigger time
                                             global origin_embedding
                                             origin_embedding = analysis['detections'][0]['embedding']
                                        
                                        # 2. Find Suspect in Current Frame Detections
                                        current_suspect_bbox = None
                                        
                                        if 'origin_embedding' in globals() and analysis.get('detections'):
                                            best_sim = 0.0
                                            # Add epsilon to avoid divide by zero
                                            norm_origin = np.linalg.norm(origin_embedding) + 1e-6
                                            ref_emb = origin_embedding / norm_origin
                                            
                                            found_high_conf_match = False
                                            
                                            for det in analysis['detections']:
                                                tid = det['track_id']
                                                curr_emb = det['embedding']
                                                norm_curr = np.linalg.norm(curr_emb) + 1e-6
                                                curr_emb = curr_emb / norm_curr
                                                sim = np.dot(ref_emb, curr_emb)
                                                
                                                # Threshold Check
                                                if sim > 0.80: # Slightly relaxed threshold for continuity
                                                    # Increment Counter for this ID
                                                    count = suspect_counters.get(tid, 0) + 1
                                                    suspect_counters[tid] = count
                                                    
                                                    # STRICT CONFIRMATION: 5 Consecutive Frames
                                                    if count >= 5:
                                                        if sim > best_sim:
                                                            best_sim = sim
                                                            current_suspect_bbox = det['bbox']
                                                            found_high_conf_match = True
                                                else:
                                                    # Reset counter for this ID if similarity drops
                                                    suspect_counters[tid] = 0
                                            
                                            # Cleaning up stale IDs could be done here but simple dict is fine for sim

                                            # 3. Match YOLO Box to Found Suspect
                                            if current_suspect_bbox:
                                                tx1, ty1, tx2, ty2 = current_suspect_bbox
                                                # Intersection
                                                ix1 = max(x1, tx1); iy1 = max(y1, ty1)
                                                ix2 = min(x2, tx2); iy2 = min(y2, ty2)
                                                if ix2 > ix1 and iy2 > iy1:
                                                    intersection = (ix2-ix1)*(iy2-iy1)
                                                    box_area = (x2-x1)*(y2-y1)
                                                    if intersection / box_area > 0.6: 
                                                        is_suspect = True
                                                        # OVERRIDE Label based on best_sim from the loop
                                                        label_text = f"SUSPECT (ORIGIN) | SIM:{best_sim:.2f} | LOCKED"

                                elif match_info:
                                    # Neighbor Camera: Use Precise Match Info
                                    target_box = match_info['bbox']
                                    target_tid = match_info['track_id']
                                    
                                    if box.id is not None and target_tid is not None:
                                        if int(box.id[0].item()) == target_tid:
                                            is_suspect = True
                                    
                                    if not is_suspect and target_box:
                                        tx1, ty1, tx2, ty2 = target_box
                                        ix1 = max(x1, tx1); iy1 = max(y1, ty1)
                                        ix2 = min(x2, tx2); iy2 = min(y2, ty2)
                                        if ix2 > ix1 and iy2 > iy1:
                                            intersection = (ix2-ix1)*(iy2-iy1)
                                            box_area = (x2-x1)*(y2-y1)
                                            if intersection / box_area > 0.6: 
                                                is_suspect = True
                                                
                                if is_suspect:
                                    color = (0, 165, 255) 
                                    eid_short = matched_event_id.split('-')[0] if matched_event_id else "TARGET"
                                    
                                    # specific label logic
                                    if cam_id == 'cam-01': 
                                        label_text = f"SUSPECT (ORIGIN)"
                                    else:
                                        # Show Similarity Score to prove ReID
                                        sim_score = match_info.get('similarity', 0.0) if match_info else 0.0
                                        if sim_score == 1.0 and match_info.get('streak', 0) > 10:
                                             # Confirmed lock
                                             label_text = f"SUSPECT | ID:{eid_short} | LOCKED"
                                        else:
                                             label_text = f"SUSPECT | ID:{eid_short} | SIM:{sim_score:.2f} | CLICK TO REJECT"
                                        
                                    # REGISTER FOR CLICK INTERACTION
                                    if cam_id not in click_registry: click_registry[cam_id] = []
                                    click_registry[cam_id].append(((x1,y1,x2,y2), tracked_id))
                            
                            # Draw BBox
                            if is_suspect:
                                # Draw Corner Brackets (Viewfinder Style)
                                l = 30 # Length of bracket
                                t = 4  # Thickness
                                # Top-Left
                                cv2.line(frame, (x1, y1), (x1+l, y1), color, t)
                                cv2.line(frame, (x1, y1), (x1, y1+l), color, t)
                                # Top-Right
                                cv2.line(frame, (x2, y1), (x2-l, y1), color, t)
                                cv2.line(frame, (x2, y1), (x2, y1+l), color, t)
                                # Bottom-Left
                                cv2.line(frame, (x1, y2), (x1+l, y2), color, t)
                                cv2.line(frame, (x1, y2), (x1, y2-l), color, t)
                                # Bottom-Right
                                cv2.line(frame, (x2, y2), (x2-l, y2), color, t)
                                cv2.line(frame, (x2, y2), (x2, y2-l), color, t)
                                
                                # Target Line to Label
                                cv2.line(frame, (x1, y1), (x1+20, y1-20), color, 2)
                                
                                # Pulsing Center Dot (Simulated)
                                cx, cy = (x1+x2)//2, (y1+y2)//2
                                cv2.circle(frame, (cx, cy), 4, color, -1)
                                
                                # Draw Label Background (High Vis)
                                t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                cv2.rectangle(frame, (x1, y1-35), (x1+t_size[0]+10, y1-5), color, -1)
                                cv2.putText(frame, label_text, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                            
                            else:
                                # Standard Box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 
                                # Standard Label
                                t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                cv2.rectangle(frame, (x1, y1-t_size[1]-5), (x1+t_size[0], y1), color, -1)
                                cv2.putText(frame, label_text, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            # --- HUD VISUALIZATION ---
            # Define Colors
            c_red = (0, 0, 255)
            c_green = (0, 255, 0)
            c_orange = (0, 165, 255)
            c_white = (255, 255, 255)
            c_black = (0, 0, 0)

            if is_system_active:
                status_color = c_orange
                if cam_id == 'cam-01':
                    status_color = c_green
                if violence_triggered and cam_id == 'cam-01':
                    status_color = c_red

                # 1. Camera Name Badge (Top Left)
                cv2.rectangle(frame, (0, 0), (140, 30), (0,0,0), -1)
                cv2.putText(frame, f"CAM-{cam_id[-2:]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_white, 1)

                # 2. Border
                cv2.rectangle(frame, (0,0), (640, 480), status_color, 2)

                # 3. Status Badge (Top Right)
                status_text = "MONITORING"
                if violence_triggered and cam_id == 'cam-01': status_text = "VIOLENCE"
                elif cam_id != 'cam-01': status_text = "TRACKING"
                
                (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (640-tw-20, 0), (640, 30), status_color, -1)
                cv2.putText(frame, status_text, (640-tw-10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_white, 2)
                
                # Active Event ID overlay (Bottom Left)
                if violence_triggered:
                     cv2.putText(frame, "EVENT: #40AE49", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                
                if int(time.time() * 2) % 2 == 0:
                     cv2.circle(frame, (620, 50), 5, c_red, -1)
                     
                # Stage 3: Summary Display (If Matched)
                # Show summary when tracking neighbors
                if cam_id != 'cam-01' and analysis.get('matched_details'):
                    md = analysis['matched_details']
                    # Retrieve the actual event to get the summary
                    active_evt = engine.active_events.get(md['event_id'])
                    if active_evt and 'summary' in active_evt.metadata:
                        summ = active_evt.metadata['summary']
                        
                        # Draw Stats Box
                        cv2.rectangle(frame, (10, 50), (200, 160), (0,0,0), -1)
                        cv2.rectangle(frame, (10, 50), (200, 160), c_orange, 1)
                        
                        cv2.putText(frame, "TARGET BIO-METRICS", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_orange, 1)
                        cv2.putText(frame, f"Avg Speed: {summ.get('posture', 'Unknown')}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_white, 1)
                        cv2.putText(frame, f"Height (Rel): {summ.get('avg_height', 0):.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_white, 1)
                        cv2.putText(frame, f"Updates: {summ.get('count', 0)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_white, 1)
                        cv2.putText(frame, f"Click box to Reject", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,255), 1)

            else:
                 # Passive Mode
                 overlay = frame.copy()
                 cv2.rectangle(overlay, (0,0), (640, 480), (0,0,0), -1)
                 frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                 cv2.putText(frame, f"CAM-{cam_id[-2:]} [OFF]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 1)

            frames[cam_id] = frame

        # Render Grid with Header
        # Add 60px Header Space
        grid_base = create_grid(frames, grid_size=(3, 3))
        h, w, c = grid_base.shape
        header_h = 80
        final_canvas = np.zeros((h + header_h, w, c), dtype=np.uint8)
        final_canvas[header_h:, :] = grid_base
        
        # Draw Header Info
        cv2.putText(final_canvas, "CONTROL ROOM: EVENT INTELLIGENCE LAYER", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(final_canvas, f"TIME: {current_time:.1f}s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Global Status
        if violence_triggered:
            status_text = "⚠️ EVENT ACTIVE - SUSPECT TRACKING IN PROGRESS"
            s_color = (0, 0, 255)
        else:
            status_text = "✅ SYSTEM NORMAL - AI SCANNING"
            s_color = (0, 255, 0)
            
        cv2.putText(final_canvas, status_text, (w - 600, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, s_color, 2)

        cv2.imshow("Control Room: Event-Driven Intelligence Layer", final_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_visual_simulation()
