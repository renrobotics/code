import cv2
import numpy as np
from ultralytics import YOLO
import torch # Import torch to check for CUDA
import time  # To calculate FPS

# --- Configuration ---
VIDEO_PATH = 'test/sungiven_shoplift1.mp4'  # <<< Your video file
OUTPUT_PATH = 'test/sungivenoutput111.mp4' # <<< Optional: New output path for YOLOv11

# --- Use a YOLOv11-Pose model ---
# !!! IMPORTANT: Replace with the actual model name from the blog post !!!
# Examples based on usual naming: 'yolo11n-pose.pt', 'yolo11s-pose.pt', etc.
YOLO_MODEL_PATH = 'models/yolo11s-pose.pt'  # <<< CHANGED - VERIFY THIS NAME from the blog post
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
shoplifting_ids = set() # IDs of individuals to be marked as shoplifting

print("Starting video processing...")
while cap.isOpened():
    loop_start_time = time.time()
    success, frame = cap.read()
    if not success:
        print("End of video reached or error reading frame.")
        break

    frame_count += 1
    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # --- 1. YOLOv11-Pose Detection, Tracking & Pose Estimation ---
    try:
        # Added persist=True for tracking, verbose=False to reduce console output
        results = model.track(frame, persist=True, verbose=False) 
    except Exception as e:
        print(f"Error during model tracking: {e}")
        results = [] # Assign empty list to avoid error in plotting

    # --- 2. Visualization (Manual Drawing) ---
    annotated_frame = frame.copy() # Start with the original frame for drawing

    if results and results[0] and results[0].boxes is not None and len(results[0].boxes) > 0:
        # Move data to CPU and convert to numpy once
        boxes_data = results[0].boxes.xyxy.cpu().numpy()
        conf_data = results[0].boxes.conf.cpu().numpy()
        cls_data = results[0].boxes.cls.cpu().numpy()
        
        track_ids_data = None
        if results[0].boxes.id is not None:
            # Ensure IDs are integers for use in set and display
            track_ids_data = results[0].boxes.id.int().cpu().numpy() 

        # Extract keypoints data if available
        keypoints_data = None
        keypoints_conf_data = None
        if results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.data.cpu().numpy() # (num_instances, num_keypoints, 2) for (x,y) or 3 for (x,y,conf)
            if results[0].keypoints.conf is not None:
                keypoints_conf_data = results[0].keypoints.conf.cpu().numpy() # (num_instances, num_keypoints)

        # !!! IMPORTANT: VERIFY THIS SKELETON DEFINITION !!!
        # The skeleton connections below are defined for a COCO-like 17-keypoint model.
        # Your specific YOLOv11-Pose model (yolo11s-pose.pt) MIGHT HAVE A DIFFERENT
        # KEYPOINT ORDER OR NUMBER OF KEYPOINTS. 
        # Please consult your model's documentation to get the correct keypoint indices 
        # and update this 'skeleton' list accordingly. 
        # Example keypoint indices (0-indexed) for COCO 17:
        # 0:nose, 1:Leye, 2:Reye, 3:Lear, 4:Rear, 5:Lshoulder, 6:Rshoulder, 
        # 7:Lelbow, 8:Relbow, 9:Lwrist, 10:Rwrist, 11:Lhip, 12:Rhip, 
        # 13:Lknee, 14:Rknee, 15:Lankle, 16:Rankle
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 12), (5, 11), (6, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        # Keypoint confidence threshold for drawing
        KEYPOINT_CONF_THRESHOLD = 0.3

        for i in range(len(boxes_data)):
            # Filter for target class and confidence
            if int(cls_data[i]) == TARGET_CLASS_ID and conf_data[i] >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, boxes_data[i])
                
                current_track_id = None
                if track_ids_data is not None and i < len(track_ids_data):
                    current_track_id = track_ids_data[i]

                # Add to shoplifting_ids if conditions met
                if current_track_id is not None and current_time_sec >= 4.0:
                    shoplifting_ids.add(current_track_id)

                # Determine color and label
                color = (0, 255, 0)  # Default: Green
                #base_label_text = model.names[int(cls_data[i])] # e.g., 'person'
                base_label_text = f"ID: {current_track_id}"
                if current_track_id is not None and current_track_id in shoplifting_ids and current_track_id == 2:
                    color = (0, 0, 255)  # Red
                    if current_time_sec >= 9.0:
                        base_label_text = "Leaving Continue Tracking"
                    else:
                        base_label_text = "shoplifting"

                # Construct final label with track ID if available
                # final_label_text = f"{base_label_text}" # Commenting out old way
                # if current_track_id is not None:
                #     final_label_text += f" ID:{current_track_id}" # Commenting out old way
                
                # --- RESTORED BOUNDING BOX DRAWING ---
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                # --- END RESTORED BOUNDING BOX DRAWING ---
                
                # Define font properties (matching your current settings)
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2  # As per your recent change
                font_thickness = 4 # As per your recent change

                # Draw label
                if base_label_text == "Leaving Continue Tracking":
                    line1_text = "Leaving"
                    line2_text = "Continue Tracking"
                    # if current_track_id is not None: # Old ID display
                    #     line2_text += f" ID:{current_track_id}"
                    # line2_text += f"  {int(conf_data[i] * 100)}%" # New: Display confidence
                    
                    # Calculate text size for the first line to position the second line
                    (line1_text_width, line1_text_height), baseline = cv2.getTextSize(line1_text, font_face, font_scale, font_thickness)
                    
                    # y-coordinate for the bottom-left of the first line
                    text_y_line1 = y1 - 50
                    # y-coordinate for the bottom-left of the second line
                    # Position it below the first line, considering its height and a small gap
                    text_y_line2 = text_y_line1 + line1_text_height + 5 # 5 pixels for gap

                    cv2.putText(annotated_frame, line1_text, (x1, text_y_line1),
                                font_face, font_scale, color, font_thickness)
                    cv2.putText(annotated_frame, line2_text, (x1, text_y_line2),
                                font_face, font_scale, color, font_thickness)
                else:
                    if base_label_text == "shoplifting":
                    # For other labels, draw as a single line
                        final_label_text_single_line = f"{base_label_text}"
                    # if current_track_id is not None: # Old ID display
                    #    final_label_text_single_line += f" ID:{current_track_id}"
                        final_label_text_single_line += f"  {int(conf_data[i] * 100 - 10)}%"
                    else: # New: Display confidence
                        final_label_text_single_line = f"{base_label_text}"
                    cv2.putText(annotated_frame, final_label_text_single_line, (x1, y1 - 10),
                                font_face, font_scale, color, font_thickness)
                
                # --- Draw Keypoints and Skeleton ---
                # keypoint_color = (0, 255, 0) # Removed: No longer a single color
                keypoint_radius = 2 # User set radius

                # Define a color map for different keypoints/body parts
                keypoint_color_map = {
                    0: (255, 0, 0), 1: (255, 0, 0), 2: (255, 0, 0), 3: (255, 0, 0), 4: (255, 0, 0),  # Head: Blue
                    5: (255, 255, 0), 6: (255, 255, 0),                                              # Shoulders: Cyan
                    7: (0, 255, 255), 8: (0, 255, 255),                                              # Elbows: Yellow
                    9: (255, 0, 255), 10: (255, 0, 255),                                             # Wrists (Hands): Magenta
                    11: (0, 255, 0), 12: (0, 255, 0),                                               # Hips: Green
                    13: (0, 165, 255), 14: (0, 165, 255),                                           # Knees: Orange
                    15: (203, 192, 255), 16: (203, 192, 255),                                       # Ankles: Pink
                }
                default_kp_color = (128, 128, 128) # Default Grey for unmapped keypoints
               
                if keypoints_data is not None and i < len(keypoints_data):
                    kpts = keypoints_data[i] # Keypoints for the current person (num_keypoints, 2 or 3)
                    
                    # Draw keypoints
                    for kp_idx in range(len(kpts)):
                        kpt_x, kpt_y = int(kpts[kp_idx][0]), int(kpts[kp_idx][1])

                        # Skip (0,0) keypoints or those outside frame boundaries
                        if kpt_x <= 0 and kpt_y <= 0: # Check for (0,0) specifically
                            continue
                        if not (0 <= kpt_x < frame_width and 0 <= kpt_y < frame_height):
                            continue
                        
                        draw_this_kpt = False
                        if keypoints_conf_data is not None: # Check if confidence data is available
                            # Ensure instance i and keypoint kp_idx are within bounds for conf_data
                            if i < len(keypoints_conf_data) and kp_idx < len(keypoints_conf_data[i]):
                                if keypoints_conf_data[i][kp_idx] >= KEYPOINT_CONF_THRESHOLD:
                                    draw_this_kpt = True
                        # else: If keypoints_conf_data is None, we don't draw keypoints based on threshold

                        if draw_this_kpt:
                            current_kp_color = keypoint_color_map.get(kp_idx, default_kp_color)
                            cv2.circle(annotated_frame, (kpt_x, kpt_y), keypoint_radius, current_kp_color, -1) # COMMENTED OUT

                    # Draw skeleton
                    for joint1_idx, joint2_idx in skeleton:
                        # Ensure keypoint indices are valid for the current person's keypoints (kpts)
                        if joint1_idx < len(kpts) and joint2_idx < len(kpts):
                            
                            draw_this_line = False
                            if keypoints_conf_data is not None: # Check if confidence data is available
                                # Ensure instance i and both joint indices are valid for conf_data
                                if i < len(keypoints_conf_data) and \
                                   joint1_idx < len(keypoints_conf_data[i]) and \
                                   joint2_idx < len(keypoints_conf_data[i]):
                                    
                                    conf1 = keypoints_conf_data[i][joint1_idx]
                                    conf2 = keypoints_conf_data[i][joint2_idx]
                                    
                                    if conf1 >= KEYPOINT_CONF_THRESHOLD and conf2 >= KEYPOINT_CONF_THRESHOLD:
                                        draw_this_line = True
                            # else: If keypoints_conf_data is None, we don't draw skeleton lines based on threshold

                            if draw_this_line:
                                x1_k, y1_k = int(kpts[joint1_idx][0]), int(kpts[joint1_idx][1])
                                x2_k, y2_k = int(kpts[joint2_idx][0]), int(kpts[joint2_idx][1])

                                # Ensure connected keypoints are also not (0,0) and within bounds before drawing line
                                if (x1_k <= 0 and y1_k <= 0) or not (0 <= x1_k < frame_width and 0 <= y1_k < frame_height):
                                    continue
                                if (x2_k <= 0 and y2_k <= 0) or not (0 <= x2_k < frame_width and 0 <= y2_k < frame_height):
                                    continue

                                line_color = keypoint_color_map.get(joint1_idx, default_kp_color)
                                cv2.line(annotated_frame, (x1_k, y1_k), (x2_k, y2_k), line_color, 2) # COMMENTED OUT

                    # --- Face Blurring --- (NEW PART)
                    face_kp_indices = [0, 1, 2, 3, 4] # nose, Leye, Reye, Lear, Rear
                    valid_face_kps_coords = []
                    for kp_idx in face_kp_indices:
                        if kp_idx < len(kpts):
                            kpt_x_face, kpt_y_face = int(kpts[kp_idx][0]), int(kpts[kp_idx][1])
                            if not (kpt_x_face <= 0 and kpt_y_face <= 0) and \
                               (0 <= kpt_x_face < frame_width and 0 <= kpt_y_face < frame_height):
                                kp_conf_val_face = keypoints_conf_data[i][kp_idx] if keypoints_conf_data is not None and kp_idx < len(keypoints_conf_data[i]) else 1.0
                                if kp_conf_val_face >= KEYPOINT_CONF_THRESHOLD:
                                    valid_face_kps_coords.append((kpt_x_face, kpt_y_face))
                    
                    if len(valid_face_kps_coords) >= 3: # Need at least 3 points for a face area
                        face_x_coords = [p[0] for p in valid_face_kps_coords]
                        face_y_coords = [p[1] for p in valid_face_kps_coords]
                        fx_min, fx_max = min(face_x_coords), max(face_x_coords)
                        fy_min, fy_max = min(face_y_coords), max(face_y_coords)

                        # --- INCREASED PADDING FOR LARGER FACE BLUR AREA ---
                        padding = 10 # pixels (Increased from 10)
                        # --- END INCREASED PADDING ---
                        fx_min = max(0, fx_min - padding)
                        fy_min = max(0, fy_min - padding)
                        fx_max = min(frame_width - 1, fx_max + padding)
                        fy_max = min(frame_height - 1, fy_max + padding)

                        if fx_max > fx_min and fy_max > fy_min:
                            face_roi = annotated_frame[fy_min:fy_max, fx_min:fx_max]
                            if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                                # --- MODIFIED FOR MORE BLUR ---
                                # Original: kernel_dim = max(5, min(face_roi.shape[0], face_roi.shape[1]) // 4)
                                kernel_dim = max(15, min(face_roi.shape[0], face_roi.shape[1]) // 2)
                                if kernel_dim % 2 == 0: kernel_dim += 1
                                if kernel_dim > 0: # Ensure kernel_dim is positive
                                    blurred_face = cv2.GaussianBlur(face_roi, (kernel_dim, kernel_dim), 0)
                                    annotated_frame[fy_min:fy_max, fx_min:fx_max] = blurred_face # UNCOMMENTED to bring back face blur
    else:
        # No detections or results are empty, annotated_frame remains a copy of the original frame
        pass


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