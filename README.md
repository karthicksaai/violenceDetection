# Violence Detection & Suspect Tracking System

This repository contains a multi-camera violence detection and suspect tracking system. It leverages **YOLO** for object detection, **OSNet** for Person Re-Identification (ReID), and a **CNN-GRU** model for violence detection. The system is designed to simulate a control room environment where events are detected, and suspects are tracked across multiple camera feeds.

## 🚀 Features

-   **Multi-Camera Simulation:** Simulates a network of cameras with overlapping fields of view.
-   **Real-time Violence Detection:** Uses a CNN-GRU model to detect violence in video streams.
-   **Suspect Tracking (ReID):** Tracks suspects across different cameras using OSNet-based feature extraction.
-   **Event Orchestration:** Manages active events, camera handoffs, and suspect locking logic.
-   **Visual Control Room:** A GUI to visualize camera feeds, alerts, and tracking status.

## 📂 Project Structure

```
.
├── visual_simulation.py        # Main entry point for the visual simulation
├── train_hockey_gru.py         # Script to train the violence detection model (Hockey Dataset)
├── evaluate_ucf_crime.py       # Script to evaluate the model on the UCF-Crime dataset
├── test_yolo_pipeline.py       # Test script for the YOLO pipeline
├── requirements.txt            # Python dependencies
├── orchestration/              # Core logic for orchestration and dispatching
│   ├── engine.py               # Orchestration Engine (Event & Camera Management)
│   ├── reid/                   # ReID module
│   │   ├── feature_extractor.py # Feature extraction using OSNet
│   │   └── osnet.py            # OSNet model definition
│   └── ...
├── violence_detection_model.pth # Pre-trained Violence Detection Model
├── yolo26n.pt                  # YOLO model weights (Custom or v8)
└── ...
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify Model Weights:**
    Ensure the following model weights are present in the root directory:
    -   `yolo26n.pt` (or `yolov8n.pt` depending on configuration)
    -   `violence_detection_model.pth`

## 🖥️ Usage

### 1. Run the Visual Simulation
This is the main demo of the system. It launches a window showing multiple camera feeds and simulates violence detection and suspect tracking.

```bash
python visual_simulation.py
```
-   **Controls:**
    -   Press `q` to exit the simulation.
    -   Click on a bounding box to provide feedback (mark as false positive).

### 2. Training the Violence Detection Model
To train the CNN-GRU model on the Hockey Fight Dataset:

```bash
python train_hockey_gru.py --dataset hockey --data_root /path/to/HockeyDataset --epochs 20
```

### 3. Evaluation
To evaluate the trained model on the UCF-Crime dataset:

```bash
python evaluate_ucf_crime.py
```
*Note: You will need the UCF-Crime dataset structure setup locally.*

### 4. Testing YOLO Pipeline
To test the object detection pipeline on a single video:

```bash
python test_yolo_pipeline.py path/to/video.mp4
```

## 🧩 System Architecture

-   **Orchestration Engine:** Manages the state of the entire camera network, handling event triggers and spatial handoffs.
-   **Dispatcher:** Processes frames using a combination of YOLO (detection), OSNet (ReID features), and GRU (activity recognition).
-   **Feature Extractor:** Extracts L2-normalized feature vectors for person re-identification.

## 📝 Configuration

-   **Camera Network:** defined in `orchestration/cameras_network.json`.
-   **Model Parameters:** logic for thresholds and model paths can be found in `orchestration/engine.py` and respective training scripts.
