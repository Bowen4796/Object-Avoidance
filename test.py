import torch
import cv2
import numpy as np
import time
import threading
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ============== CONFIGURATION PARAMETERS ==============
# Model settings
MODEL_ID = "IDEA-Research/grounding-dino-base"
TEXT_QUERIES = "a bottle. a glasses. a pillow."  # Objects to detect (lowercase, dot-separated)
TEXT_THRESHOLD = 0.1  # Confidence threshold for detection

# Inference settings
INFERENCE_INTERVAL = 1  # Run inference every N seconds
INFERENCE_FRAME_WIDTH = 640  # Width to resize frame for inference
INFERENCE_FRAME_HEIGHT = 480  # Height to resize frame for inference

# Display settings
DISPLAY_THICKNESS = 2  # Bounding box line thickness
DISPLAY_FONT_SCALE = 0.5  # Label font size
DISPLAY_COLOR = (0, 255, 0)  # BGR color for boxes (green)

# Camera settings
CAMERA_INDEX = 0  # Camera device index (0 = default camera)
# =====================================================

model_id = MODEL_ID
# Use Metal Performance Shaders on M-chip Macs
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Metal Performance Shaders (MPS) on Apple Silicon")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA")
else:
    device = "cpu"
    print("Using CPU (slow)")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_QUERIES

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Camera opened. Press 'q' to quit.")

last_inference_time = 0
current_boxes = []
current_labels = []
current_scores = []
inference_lock = threading.Lock()
stop_inference = False

def run_inference(frame_rgb, original_height, original_width):
    """Run inference in background thread"""
    global current_boxes, current_labels, current_scores
    
    resized_frame = cv2.resize(frame_rgb, (INFERENCE_FRAME_WIDTH, INFERENCE_FRAME_HEIGHT))
    pil_image = Image.fromarray(resized_frame)
    
    inputs = processor(images=pil_image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(INFERENCE_FRAME_HEIGHT, INFERENCE_FRAME_WIDTH)]  # Use resized dimensions for target
    )
    
    # Thread-safe update
    with inference_lock:
        boxes = results[0]["boxes"].cpu().numpy()
        # Scale boxes back to original frame size
        scale_x = original_width / INFERENCE_FRAME_WIDTH
        scale_y = original_height / INFERENCE_FRAME_HEIGHT
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
        boxes[:, [1, 3]] *= scale_y  # Scale y coordinates
        current_boxes = boxes
        current_labels = results[0]["labels"]
        current_scores = results[0]["scores"].cpu().numpy()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read frame")
        break
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width = frame.shape[:2]
    
    # Start inference in background thread every 2.5 seconds
    current_time = time.time()
    if current_time - last_inference_time >= INFERENCE_INTERVAL:
        last_inference_time = current_time
        # Run inference in separate thread so it doesn't block display
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
    cv2.imshow('Grounding DINO - Object Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
