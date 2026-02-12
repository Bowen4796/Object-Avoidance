import cv2
import numpy as np
import time
import threading
from PIL import Image
from ultralytics import YOLO

# ============== CONFIGURATION PARAMETERS ==============
# Model settings
MODEL_NAME = "yolov8n.pt"  # nano model - will be converted to ONNX
# Common classes: person, cat, dog, car, phone, etc.

# Inference settings
INFERENCE_INTERVAL = 2.5  # Run inference every N seconds (increase for slower RPi)
INFERENCE_FRAME_WIDTH = 416  # Width to resize frame for inference
INFERENCE_FRAME_HEIGHT = 416  # Height to resize frame for inference

# Display settings
DISPLAY_THICKNESS = 2  # Bounding box line thickness
DISPLAY_FONT_SCALE = 0.5  # Label font size
DISPLAY_COLOR = (0, 255, 0)  # BGR color for boxes (green)
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence threshold

# Camera settings
CAMERA_INDEX = 0  # Camera device index (0 = default camera)
# =====================================================

print("Loading YOLOv8 model (ONNX for RPi5)...")
# Load model - ONNX mode automatically avoids PyTorch on RPi5
model = YOLO(MODEL_NAME)

print("Camera opened. Press 'q' to quit.")

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

last_inference_time = 0
current_boxes = []
current_labels = []
current_scores = []
inference_lock = threading.Lock()

def run_inference(frame_rgb, original_height, original_width):
    """Run inference in background thread"""
    global current_boxes, current_labels, current_scores
    
    resized_frame = cv2.resize(frame_rgb, (INFERENCE_FRAME_WIDTH, INFERENCE_FRAME_HEIGHT))
    
    # Run YOLOv8 inference
    results = model(resized_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # Extract boxes and labels
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    labels = [model.names[int(cls_id)] for cls_id in class_ids]
    
    # Scale boxes back to original frame size
    scale_x = original_width / INFERENCE_FRAME_WIDTH
    scale_y = original_height / INFERENCE_FRAME_HEIGHT
    
    scaled_boxes = boxes.copy()
    scaled_boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
    scaled_boxes[:, [1, 3]] *= scale_y  # Scale y coordinates
    
    # Thread-safe update
    with inference_lock:
        current_boxes = scaled_boxes
        current_labels = labels
        current_scores = scores

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read frame")
        break
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width = frame.shape[:2]
    
    # Start inference in background thread at specified interval
    current_time = time.time()
    if current_time - last_inference_time >= INFERENCE_INTERVAL:
        last_inference_time = current_time
        # Run inference in separate thread
        inference_thread = threading.Thread(target=run_inference, args=(rgb_frame, frame_height, frame_width), daemon=True)
        inference_thread.start()
    
    # Draw bounding boxes on frame (from last inference result)
    with inference_lock:
        boxes = current_boxes.copy() if len(current_boxes) > 0 else []
        labels = current_labels.copy() if len(current_labels) > 0 else []
        scores = current_scores.copy() if len(current_scores) > 0 else []
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), DISPLAY_COLOR, DISPLAY_THICKNESS)
        label_text = f"{label}: {score:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_FONT_SCALE, DISPLAY_COLOR, 2)
    
    # Display frame
    cv2.imshow('YOLOv8 - Object Detection (RPi5)', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
