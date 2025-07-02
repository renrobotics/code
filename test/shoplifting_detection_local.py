import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import collections

# --- Configuration ---
VIDEO_PATH = 'sungivenshoplift1.mp4'  # <<< --- YOUR SHOPLIFTING VIDEO FILE
OUTPUT_VIDEO_PATH = 'outputtest.mp4'  # Optional
YOLO_MODEL_PATH = '..\yolo11s-pose.pt'  # Or your .pt pose model, e.g., 'yolov8n-pose.pt'

TARGET_CLASS_ID = 0  # COCO class ID for 'person'
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence for YOLO

# Behavior detection parameters
SUSPICIOUS_ZONE_THRESHOLD_FRAMES = 10  # e.g., if hand in zone for ~0.5s at 30fps, or ~1.5s at 10fps
WRIST_HISTORY_LEN = 30  # Keep history for this many frames for each person

# Keypoint indices (COCO format, typical for YOLO-Pose)
# Verify these if your model uses a different keypoint convention
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

# --- Initialize Device ---
if torch.cuda.is_available():
    print("[INFO] CUDA is available! Using GPU.")
    device = 'cuda'
else:
    print("[INFO] Warning: CUDA not available. Running on CPU.")
    device = 'cpu'

# --- Initialize Model ---
print(f"[INFO] Loading YOLO-Pose model: {YOLO_MODEL_PATH}...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    # model.to(device) # For .pt models. .engine files are already device-specific.
    # YOLO() class should handle device for .engine.
    print("[INFO] YOLO-Pose model loaded successfully.")
    if hasattr(model, 'device'):
        print(f"[INFO] Model effectively on device: {model.device}")
except Exception as e:
    print(f"[ERROR] Error loading YOLO model: {e}")
    exit()

# --- Video I/O Setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Video: {VIDEO_PATH}, {frame_width}x{frame_height} @ {fps:.2f} FPS")

writer = None
if OUTPUT_VIDEO_PATH:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"[INFO] Output video will be saved to: {OUTPUT_VIDEO_PATH}")

# --- Data Structures for Behavior Analysis ---
# Stores data per track_id: {'in_zone_counter_left': 0, 'in_zone_counter_right': 0, 'is_suspicious': False, 'alert_triggered': False}
person_behavior_data = collections.defaultdict(lambda: {
    'in_zone_counter_left': 0,
    'in_zone_counter_right': 0,
    'is_suspicious': False,
    'alert_triggered_left': False,  # To avoid repeated alerts for the same event
    'alert_triggered_right': False
})

# --- Processing Loop ---
frame_num = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[INFO] End of video or error reading frame.")
        break
    frame_num += 1
    start_time = time.time()

    # --- 1. YOLO Detection, Tracking & Pose Estimation ---
    # Use persist=True for tracking across frames
    results = model.track(frame, persist=True, classes=[TARGET_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)

    annotated_frame = frame.copy()  # Start with a copy for drawing

    if results and results[0]:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id
        keypoints_data = results[0].keypoints.data.cpu().numpy()  # (num_persons, num_keypoints, xy_or_xyc)
        # confidences = results[0].boxes.conf.cpu().numpy() # If needed

        if track_ids is not None:  # Ensure tracks are present
            track_ids = track_ids.int().tolist()

            for i, track_id in enumerate(track_ids):
                person_box = boxes[i]
                person_kps = keypoints_data[i]  # Keypoints for this specific person

                x1, y1, x2, y2 = map(int, person_box)
                # Draw basic bounding box and track ID (YOLO's plot does this better, but for custom logic)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw pose using Ultralytics built-in plot for this person
                # To do this, we create a temporary results-like object for a single person
                # This is a bit of a hack if results[0].plot() plots all.
                # A cleaner way would be to manually draw keypoints if needed.
                # For now, let's assume results[0].plot() will draw all skeletons correctly.

                # --- Behavior Analysis for this person (track_id) ---
                behavior_data = person_behavior_data[track_id]


                # Get key keypoints (ensure they are detected, i.e., conf > 0 or x,y > 0)
                # Keypoint format from Ultralytics is usually [x, y, confidence]
                def get_kp(kp_idx):
                    if kp_idx < len(person_kps) and person_kps[kp_idx][2] > 0.1:  # Check confidence
                        return int(person_kps[kp_idx][0]), int(person_kps[kp_idx][1])
                    return None


                l_wrist_pt = get_kp(L_WRIST)
                r_wrist_pt = get_kp(R_WRIST)
                l_hip_pt = get_kp(L_HIP)
                r_hip_pt = get_kp(R_HIP)
                l_shoulder_pt = get_kp(L_SHOULDER)
                r_shoulder_pt = get_kp(R_SHOULDER)

                # Define suspicious zone (simple torso area below shoulders, between hips)
                # This is a very basic heuristic and needs refinement for real-world use.
                suspicious_zone_y_start = min(l_shoulder_pt[1] if l_shoulder_pt else y1,
                                              r_shoulder_pt[1] if r_shoulder_pt else y1)
                suspicious_zone_y_end = max(l_hip_pt[1] if l_hip_pt else y2,
                                            r_hip_pt[1] if r_hip_pt else y2)
                suspicious_zone_x_start = min(l_hip_pt[0] if l_hip_pt else x1,
                                              r_hip_pt[0] if r_hip_pt else x1,
                                              # More robust: min(l_shoulder_x, r_shoulder_x)
                                              l_shoulder_pt[0] if l_shoulder_pt else x1,
                                              r_shoulder_pt[0] if r_shoulder_pt else x1)
                suspicious_zone_x_end = max(l_hip_pt[0] if l_hip_pt else x2,
                                            r_hip_pt[0] if r_hip_pt else x2,
                                            # More robust: max(l_shoulder_x, r_shoulder_x)
                                            l_shoulder_pt[0] if l_shoulder_pt else x2,
                                            r_shoulder_pt[0] if r_shoulder_pt else x2)

                # Ensure zone has valid dimensions
                if suspicious_zone_y_start < suspicious_zone_y_end and suspicious_zone_x_start < suspicious_zone_x_end:
                    # For visualization of the suspicious zone (optional)
                    # cv2.rectangle(annotated_frame, (suspicious_zone_x_start, suspicious_zone_y_start),
                    #               (suspicious_zone_x_end, suspicious_zone_y_end), (255, 0, 255), 1)

                    # Check left wrist
                    if l_wrist_pt:
                        if (suspicious_zone_x_start < l_wrist_pt[0] < suspicious_zone_x_end and
                                suspicious_zone_y_start < l_wrist_pt[1] < suspicious_zone_y_end):
                            behavior_data['in_zone_counter_left'] += 1
                        else:
                            behavior_data['in_zone_counter_left'] = 0  # Reset if hand moves out
                            behavior_data['alert_triggered_left'] = False  # Reset alert

                    # Check right wrist
                    if r_wrist_pt:
                        if (suspicious_zone_x_start < r_wrist_pt[0] < suspicious_zone_x_end and
                                suspicious_zone_y_start < r_wrist_pt[1] < suspicious_zone_y_end):
                            behavior_data['in_zone_counter_right'] += 1
                        else:
                            behavior_data['in_zone_counter_right'] = 0  # Reset
                            behavior_data['alert_triggered_right'] = False

                # Check for suspicious duration
                current_suspicion = False
                alert_message = ""
                if behavior_data['in_zone_counter_left'] > SUSPICIOUS_ZONE_THRESHOLD_FRAMES:
                    current_suspicion = True
                    if not behavior_data['alert_triggered_left']:
                        alert_message += f"Suspicious: L_Hand in zone (ID {track_id})!\n"
                        print(
                            f"[ALERT] Frame {frame_num}: Track ID {track_id} - Left hand in suspicious zone for {behavior_data['in_zone_counter_left']} frames.")
                        behavior_data['alert_triggered_left'] = True  # Trigger alert only once per continuous event
                        # TODO: Start buffering frames for saving here

                if behavior_data['in_zone_counter_right'] > SUSPICIOUS_ZONE_THRESHOLD_FRAMES:
                    current_suspicion = True
                    if not behavior_data['alert_triggered_right']:
                        alert_message += f"Suspicious: R_Hand in zone (ID {track_id})!"
                        print(
                            f"[ALERT] Frame {frame_num}: Track ID {track_id} - Right hand in suspicious zone for {behavior_data['in_zone_counter_right']} frames.")
                        behavior_data['alert_triggered_right'] = True
                        # TODO: Start buffering frames for saving here

                behavior_data['is_suspicious'] = current_suspicion

                if behavior_data['is_suspicious']:
                    # Change box color for suspicious person
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated_frame, "SUSPICIOUS", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if alert_message:  # To avoid re-printing on every frame while suspicious
                        print(alert_message.strip())  # Print consolidated alert

        # Use Ultralytics plot() for skeletons if available (it plots all)
        # This will draw over our custom boxes if we don't clear them,
        # or we can choose to draw skeletons manually.
        # For simplicity, let's let it draw everything.
        # The `annotated_frame` already contains our custom boxes/text.
        # We want to add poses to this.
        if results and results[0] and results[0].keypoints is not None:
            annotated_frame = results[0].plot(boxes=False, labels=False,
                                              img=annotated_frame)  # Draw only poses on our annotated_frame

    # Calculate and display FPS
    processing_time = time.time() - start_time
    current_fps = 1.0 / processing_time if processing_time > 0 else 0
    cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Shoplifting Detection Test", annotated_frame)

    # Write frame to output video
    if writer:
        writer.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

# --- Release Resources ---
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("[INFO] Done.")