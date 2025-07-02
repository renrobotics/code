import cv2
import numpy as np
from ultralytics import YOLO
import torch # Import torch to check for CUDA
import time  # To calculate FPS

# --- Configuration ---
VIDEO_PATH = 'sungivenshoplift1.mp4'  # <<< Your video file
OUTPUT_PATH = 'outputtest.mp4' # <<< Optional: New output path for YOLOv11

# --- Use a YOLOv11-Pose model ---
# !!! IMPORTANT: Replace with the actual model name from the blog post !!!
# Examples based on usual naming: 'yolo11n-pose.pt', 'yolo11s-pose.pt', etc.
YOLO_MODEL_PATH = 'best.pt'  # <<< CHANGED - VERIFY THIS NAME from the blog post
TARGET_CLASS_ID = 0  # COCO class ID for 'person'
CONFIDENCE_THRESHOLD = 0.5  # Minimum detection confidence for bounding box

# --- Check GPU Availability ---
if torch.cuda.is_available():
    # Explicitly check PyTorch version compatibility if needed, although usually handled by ultralytics install
    print("CUDA is available! Using GPU.")
    device = 'cuda'
else:
    print("Warning: CUDA not available. Running on CPU.")
    device = 'cpu'

# --- Initialize Models ---
print(f"Loading YOLOv11-Pose model: {YOLO_MODEL_PATH}...")
# Load the YOLOv11-Pose model (assuming API compatibility)
try:
    model = YOLO(YOLO_MODEL_PATH)
    model.to(device) # Move model to GPU if available
    print("YOLOv11-Pose model loaded successfully.")
except Exception as e:
     print(f"Error loading YOLO model: {e}")
     print(f"Please ensure the model file '{YOLO_MODEL_PATH}' exists, is a valid name,")
     print("and you have the latest Ultralytics library supporting YOLOv11.")
     exit()

# --- Video I/O Setup ---
print(f"Opening video file: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

writer = None
if OUTPUT_PATH:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    if writer.isOpened():
        print(f"Output video will be saved to: {OUTPUT_PATH}")
    else:
        writer = None
        print(f"Error: Could not open video writer for path: {OUTPUT_PATH}")

# --- Processing Loop ---
frame_count = 0
total_time = 0.0 # For average FPS calculation

print("Starting video processing...")
while cap.isOpened():
    loop_start_time = time.time()
    success, frame = cap.read()
    if not success:
        print("End of video reached or error reading frame.")
        break

    frame_count += 1

    # --- 1. YOLOv11-Pose Detection, Tracking & Pose Estimation ---
    # Assuming model.track() works similarly for v11 pose models
    try:
        results = model.track(frame)
    except Exception as e:
        print(f"Error during model tracking: {e}")
        # Decide how to handle errors - skip frame, break loop etc.
        # For now, just print and try to continue with an empty results list
        results = [] # Assign empty list to avoid error in plotting

    # --- 2. Visualization ---
    # Assuming results[0].plot() works similarly for v11 pose models
    if results and results[0]: # Check if results are not empty and contain data
         # plot() draws boxes, track IDs (if available), class labels, and poses
         annotated_frame = results[0].plot(boxes=True, labels=True)
    else:
         annotated_frame = frame.copy() # No results, use original frame


    # Calculate and display FPS for this frame
    loop_end_time = time.time()
    frame_time = loop_end_time - loop_start_time
    processing_fps = 1.0 / frame_time if frame_time > 0 else 0
    total_time += frame_time

    cv2.putText(annotated_frame, f"FPS: {processing_fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the annotated frame
    cv2.imshow("YOLOv11-Pose Processed Video", annotated_frame) # Updated window title

    # Write frame to output video
    if writer:
        writer.write(annotated_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# --- Release Resources ---
print("Releasing resources...")
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

# Calculate and print average FPS
if frame_count > 0:
    average_fps = frame_count / total_time
    print(f"Processing finished. Average FPS: {average_fps:.2f}")
else:
    print("Processing finished. No frames were processed.")

print("Done.")