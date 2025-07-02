# main.py
"""
Main application script for multi-camera pose estimation.
Orchestrates camera streams, pose estimation (multi-threaded on RESIZED frames), and display.
"""
import cv2
import numpy as np
import time
import threading
import queue # Ensure queue is imported
import config # Import configuration
from camera_streamer import CameraStreamer # Import the streamer class
from pose_estimator import PoseEstimator # Import the estimator class

# --- Inference Worker Thread Function ---
# (Keep the inference_worker function as it was in the previous correct version)
def inference_worker(worker_id, input_queue, output_queue, estimator, stop_event):
    """
    Thread function to continuously process frames from the input queue
    (resizing them first) and put results into the output queue.
    """
    print(f"[Worker-{worker_id}] Starting...")
    while not stop_event.is_set():
        try:
            # Get a task (camera_idx, raw_frame) from the input queue
            camera_idx, raw_frame = input_queue.get(block=True, timeout=1)

            if raw_frame is None:
                 input_queue.task_done()
                 continue

            # --- Resize the frame BEFORE inference ---
            resized_frame = cv2.resize(raw_frame, (config.SMALL_WIDTH, config.SMALL_HEIGHT))

            # --- Run Pose Estimation on the RESIZED frame ---
            results = estimator.predict(resized_frame) # Pass resized frame to model

            # --- Get Annotated Frame (plotting on the RESIZED frame) ---
            annotated_resized_frame = estimator.plot(results, resized_frame)

            # --- Put result (camera_idx, annotated_resized_frame) into output queue ---
            try:
                 output_queue.put((camera_idx, annotated_resized_frame), block=True, timeout=1)
            except queue.Full:
                 print(f"[Worker-{worker_id}] Output queue full. Discarding result for Cam {camera_idx}")

            input_queue.task_done() # Signal that this task is complete

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker-{worker_id}] Error: {type(e).__name__}: {e}")
            # Ensure task_done is called even if processing fails, to prevent input_queue.join() from hanging
            # Be careful if the source of the error means the task wasn't truly "done"
            try:
                input_queue.task_done()
            except ValueError: # task_done called too many times
                pass


    print(f"[Worker-{worker_id}] Stopping...")


# --- Main Execution ---
def main():
    print("[Main] Starting application...")

    stop_event = threading.Event()
    BUFFER_SECONDS = 30 # Allow up to 30 seconds buffer
# Calculate approximate input rate based on config
    approx_input_fps_per_cam = config.FFMPEG_FPS_OUTPUT
    total_input_fps = len(config.CAMERA_URLS) * approx_input_fps_per_cam
    queue_size = total_input_fps * BUFFER_SECONDS 
# Add some extra headroom
    input_queue_maxsize = max(queue_size + 50, 200) # Ensure a reasonable minimum size
    output_queue_maxsize = input_queue_maxsize # Keep output similar for now

    input_frame_queue = queue.Queue(maxsize=input_queue_maxsize)
    output_frame_queue = queue.Queue(maxsize=output_queue_maxsize)
    print(f"[Main] Setting Queue Maxsize: ~{input_queue_maxsize}")
    # # --- Initialize Queues ---
    # input_frame_queue = queue.Queue(maxsize=len(config.CAMERA_URLS) * 2)
    # # *** ADDED THE MISSING LINE BELOW ***
    # output_frame_queue = queue.Queue(maxsize=len(config.CAMERA_URLS) * 2) # Output queue for (idx, annotated_frame)

    # --- Initialize Pose Estimator ---
    try:
        estimator = PoseEstimator()
    except Exception as e:
        print(f"[Main] Failed to initialize Pose Estimator: {e}. Exiting.")
        return

    # --- Initialize and start Camera Streamers ---
    streamers = []
    camera_threads = []
    print("[Main] Starting camera streams...")
    for idx, url in enumerate(config.CAMERA_URLS):
        streamer = CameraStreamer(idx, url, input_frame_queue, stop_event)
        streamer.start()
        streamers.append(streamer)
        camera_threads.append(streamer.thread)
        time.sleep(0.1)

    # --- Initialize and start Inference Workers ---
    inference_threads = []
    print(f"[Main] Starting {config.NUM_INFERENCE_WORKERS} inference workers...")
    for i in range(config.NUM_INFERENCE_WORKERS):
        # Make sure output_frame_queue is passed here
        worker = threading.Thread(target=inference_worker,
                                  args=(i, input_frame_queue, output_frame_queue, estimator, stop_event),
                                  daemon=True)
        worker.start()
        inference_threads.append(worker)

    # --- Display Setup ---
    num_cams = len(config.CAMERA_URLS)
    if num_cams == 8: grid_cols, grid_rows = 4, 2
    elif num_cams == 6: grid_cols, grid_rows = 3, 2
    else: grid_cols = int(np.ceil(np.sqrt(num_cams))); grid_rows = int(np.ceil(num_cams / grid_cols))
    print(f"[Main] Display layout: {grid_rows} rows x {grid_cols} columns")
    print("[Main] Starting display loop. Press 'q' to quit.")
    print("[Main] Note: AI inference runs on resized frames in parallel threads.")

    latest_annotated_frames = {i: None for i in range(num_cams)}
    last_display_time = time.time()
    display_interval = 1.0 / config.DISPLAY_FPS_TARGET

    # --- Main Display Loop (Corrected) ---
    while not stop_event.is_set():
        current_time = time.time()
        # --- Get results from output queue ---
        while not output_frame_queue.empty():
             try:
                 # Get results without blocking display for too long
                 # Use the correct variable name now defined: output_frame_queue
                 cam_idx, annotated_frame = output_frame_queue.get_nowait()
                 latest_annotated_frames[cam_idx] = annotated_frame
                 # *** REMOVED incorrect output_queue.task_done() call ***
             except queue.Empty:
                 break # No more items for now

        # --- Update display periodically ---
        if (current_time - last_display_time) < display_interval:
            time.sleep(0.01)
            continue
        last_display_time = current_time

        all_rows = []
        grid_processed_count = 0

        try:
            grid_start_time = time.time()
            for r in range(grid_rows):
                row_frames = []
                for c in range(grid_cols):
                    idx = r * grid_cols + c
                    placeholder_text = f"CAM {idx}"
                    status_text = "No Signal"

                    if idx < num_cams:
                        # Get the latest *processed and already resized* frame
                        small_frame = latest_annotated_frames.get(idx)

                        if small_frame is not None:
                             grid_processed_count +=1
                             # No resize needed here
                        else:
                            # Create Placeholder
                            small_frame = np.zeros((config.SMALL_HEIGHT, config.SMALL_WIDTH, 3), dtype=np.uint8)
                            try:
                                if idx < len(streamers) and not streamers[idx].is_alive() and not stop_event.is_set(): status_text = "Reader Err"
                                elif small_frame is None and streamers[idx].is_alive(): status_text = "Waiting..."
                            except IndexError: status_text = "Unknown State"
                            cv2.putText(small_frame, placeholder_text, (10, config.SMALL_HEIGHT // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(small_frame, status_text, (10, config.SMALL_HEIGHT // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        # Empty grid cell
                        small_frame = np.zeros((config.SMALL_HEIGHT, config.SMALL_WIDTH, 3), dtype=np.uint8)
                        cv2.putText(small_frame, "EMPTY", (10, config.SMALL_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

                    row_frames.append(small_frame)

                if row_frames:
                    current_row_image = np.hstack(row_frames)
                    all_rows.append(current_row_image)

            # --- Display Logic ---
            if all_rows:
                merged_frame = np.vstack(all_rows)
                grid_end_time = time.time()
                grid_fps = 1.0 / (grid_end_time - grid_start_time) if (grid_end_time - grid_start_time) > 0 else 0
                cv2.putText(merged_frame, f"Disp FPS: {grid_fps:.2f} (Inf Q: {input_frame_queue.qsize()}, Out Q: {output_frame_queue.qsize()})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(config.DISPLAY_WINDOW_NAME, merged_frame)
            elif not stop_event.is_set():
                 placeholder_display = np.zeros((config.SMALL_HEIGHT * grid_rows, config.SMALL_WIDTH * grid_cols, 3), dtype=np.uint8)
                 cv2.putText(placeholder_display, "Initializing Streams...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                 cv2.imshow(config.DISPLAY_WINDOW_NAME, placeholder_display)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Main] 'q' pressed. Stopping...")
                stop_event.set()
                break

        except KeyboardInterrupt:
            print("[Main] Ctrl+C detected. Stopping...")
            stop_event.set()
            break
        except Exception as e:
            print(f"[Main] Error in display loop: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    # --- Cleanup ---
    # (Cleanup code remains the same)
    print("[Main] Exiting display loop. Signaling stop to threads...")
    stop_event.set()
    time.sleep(2)
    print("[Main] Waiting briefly for threads to stop...")
    time.sleep(1)
    print("[Main] Closing OpenCV windows.")
    cv2.destroyAllWindows()
    print("[Main] Application finished.")

if __name__ == "__main__":
    main()