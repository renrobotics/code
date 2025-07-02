import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict, deque

# --- Configuration (User should set these) ---
VIDEO_PATH = "activity.mp4"  # <<< Replace with your video path
OUTPUT_PATH = "activityoutput.mp4"  # <<< Optional: Replace or set to None
YOLO_MODEL_PATH = "../models/yolov8n.pt"  # Or yolov8n-pose.pt, etc.

# --- Constants for Activity Detection ---
TARGET_CLASS_ID = 0  # COCO class ID for 'person'
CONFIDENCE_THRESHOLD = 0.5  # Minimum detection confidence for YOLO

MOVEMENT_THRESHOLD = 5       # Min pixel change in bbox center to be considered 'walking'
STANDING_FRAMES_THRESHOLD = 5 # Frames below movement_threshold to be 'standing'
WALKING_FRAMES_THRESHOLD = 2  # Frames above movement_threshold to be 'walking'
MAX_HISTORY = 10             # Max number of past center points to store per track_id

def get_bbox_center(bbox):
    """Calculates the center of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def main():
    print(f"Using video: {VIDEO_PATH}")
    if OUTPUT_PATH:
        print(f"Output will be saved to: {OUTPUT_PATH}")
    print(f"Using YOLO model: {YOLO_MODEL_PATH}")

    # --- Check GPU Availability ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Initialize YOLO Model ---
    try:
        model = YOLO(YOLO_MODEL_PATH)
        model.to(device)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # --- Video I/O Setup ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    writer = None
    if OUTPUT_PATH:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
        if not writer.isOpened():
            print(f"Error opening video writer for {OUTPUT_PATH}")
            writer = None # Proceed without writing

    # --- Data Structures for Tracking Activity ---
    # Stores recent center positions for each track_id: {track_id: deque([(x,y), (x,y), ...])}
    track_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
    # Stores current activity state: {track_id: "standing" or "walking"}
    track_activity_state = defaultdict(lambda: "standing") # Default to standing
    # Stores counters for consecutive frames in a state: {track_id: count}
    track_standing_counter = defaultdict(int)
    track_walking_counter = defaultdict(int)

    # --- Processing Loop ---
    frame_count = 0
    print("Starting video processing for activity detection...")

    # --- Kernel for blurring ---
    blur_kernel_size = (51, 51) # Adjust for more or less blur

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break

        frame_count += 1
        annotated_frame = frame.copy()

        # Create a blurred version of the entire frame
        blurred_frame_full = cv2.GaussianBlur(frame, blur_kernel_size, 0)
        # Create a mask for the foreground (people)
        foreground_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        try:
            results = model.track(source=frame, persist=True, classes=[TARGET_CLASS_ID], conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)
        except Exception as e:
            print(f"Error during model tracking: {e}")
            results = []

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_data = results[0].boxes.xyxy.cpu().numpy()
            track_ids_data = results[0].boxes.id
            
            current_frame_track_ids = set()

            if track_ids_data is not None:
                track_ids_data = track_ids_data.int().cpu().numpy()
                for i in range(len(boxes_data)):
                    if i >= len(track_ids_data): continue # Should not happen if data is consistent

                    track_id = track_ids_data[i]
                    current_frame_track_ids.add(track_id)
                    bbox = boxes_data[i]
                    x1, y1, x2, y2 = map(int, bbox)
                    center_x, center_y = get_bbox_center(bbox)

                    activity_label = "unknown"
                    color = (128, 128, 128) # Default Grey for unknown

                    # Update history
                    if len(track_history[track_id]) > 0:
                        prev_center_x, prev_center_y = track_history[track_id][-1]
                        movement = np.sqrt((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)
                        
                        if movement < MOVEMENT_THRESHOLD:
                            track_standing_counter[track_id] += 1
                            track_walking_counter[track_id] = 0 # Reset other counter
                            if track_standing_counter[track_id] >= STANDING_FRAMES_THRESHOLD:
                                track_activity_state[track_id] = "standing"
                        else:
                            track_walking_counter[track_id] += 1
                            track_standing_counter[track_id] = 0 # Reset other counter
                            if track_walking_counter[track_id] >= WALKING_FRAMES_THRESHOLD:
                                track_activity_state[track_id] = "walking"
                    else:
                        # Not enough history to determine movement, assume standing or initial state
                        track_activity_state[track_id] = "standing" # Or some initial label
                        track_standing_counter[track_id] = 1 # Start counting towards standing

                    track_history[track_id].append((center_x, center_y))
                    activity_label = track_activity_state[track_id]

                    # Set color based on activity
                    if activity_label == "standing":
                        color = (0, 255, 0)  # Green
                    elif activity_label == "walking":
                        color = (0, 0, 255)  # Red
                    
                    # --- Logic to include label in foreground mask ---
                    # Calculate text properties for defining the mask area accurately.
                    current_label_text_for_mask = f"ID:{track_id} {activity_label}"
                    font_face_for_mask = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale_for_mask = 0.7
                    font_thickness_for_mask = 2
                    text_padding_for_mask = 5 # Padding around the text area in the mask

                    (text_w_mask, text_h_mask), baseline_mask = cv2.getTextSize(current_label_text_for_mask, font_face_for_mask, font_scale_for_mask, font_thickness_for_mask)

                    # The text label is drawn starting at (x1, y1 - 10) on annotated_frame.
                    # y1 - 10 is the baseline of the text.
                    text_baseline_y_on_frame = y1 - 10 

                    # Define the rectangle for the foreground_mask.
                    # This rectangle should encompass both the original bounding box (x1,y1,x2,y2)
                    # and the text label area.
                    mask_rect_x1 = x1
                    mask_rect_y1 = text_baseline_y_on_frame - text_h_mask - text_padding_for_mask # Top of the text area
                    mask_rect_x2 = max(x2, x1 + text_w_mask + text_padding_for_mask) # Furthest x extent of box or text
                    mask_rect_y2 = y2 # Bottom of the original bounding box

                    # Ensure coordinates are within frame boundaries and are valid.
                    mask_rect_x1 = max(0, mask_rect_x1)
                    mask_rect_y1 = max(0, mask_rect_y1)
                    mask_rect_x2 = min(frame_width - 1, mask_rect_x2) # frame_width is from outer scope
                    mask_rect_y2 = min(frame_height - 1, mask_rect_y2) # frame_height is from outer scope

                    if mask_rect_x1 < mask_rect_x2 and mask_rect_y1 < mask_rect_y2:
                        cv2.rectangle(foreground_mask,
                                      (int(mask_rect_x1), int(mask_rect_y1)),
                                      (int(mask_rect_x2), int(mask_rect_y2)),
                                      (255), -1) # White, filled
                    else:
                        # If the calculated mask region is invalid (e.g. text too large, box too small/at edge),
                        # fall back to just the bounding box for the mask to avoid errors.
                        # This ensures at least the primary object is not blurred.
                        cv2.rectangle(foreground_mask, (x1, y1), (x2, y2), (255), -1)

                    # --- End of label mask logic ---
                    
                    # Draw bounding box on the annotated_frame (this is existing code)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    # Draw label (this is existing code)
                    label_text = f"ID:{track_id} {activity_label}" # This redefinition is fine as it's used by putText
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Clean up data for tracks that are no longer present
            for track_id_to_remove in list(track_history.keys()):
                if track_id_to_remove not in current_frame_track_ids:
                    del track_history[track_id_to_remove]
                    del track_activity_state[track_id_to_remove]
                    del track_standing_counter[track_id_to_remove]
                    del track_walking_counter[track_id_to_remove]

        # Combine the blurred background with the sharp foreground
        # Convert foreground_mask to 3 channels to be compatible with frame
        foreground_mask_3ch = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)
        
        # Where mask is white, use original frame, otherwise use blurred frame
        final_frame = np.where(foreground_mask_3ch == (255, 255, 255), annotated_frame, blurred_frame_full)

        cv2.imshow("Activity Detection", final_frame)
        if writer:
            writer.write(final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting by user request.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Processing finished.")

if __name__ == "__main__":
    if VIDEO_PATH == "path/to/your/video.mp4" or not VIDEO_PATH:
        print("\n" + "="*50)
        print("IMPORTANT: Please open test/activity.py and set the VIDEO_PATH variable")
        print("to your actual video file path. You can also set OUTPUT_PATH.")
        print("="*50 + "\n")
    else:
        main()
