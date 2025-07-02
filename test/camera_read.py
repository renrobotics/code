import os
import cv2 # Still needed for display (imshow, resize, putText, etc.)
import threading
import numpy as np
import time
import ffmpeg # Import ffmpeg-python
import subprocess # Needed to interact with the process

# --- Configuration ---
camera_urls = [
    # "rtsp://admin:SFC@dmin2022@24.109.42.130:554/Streaming/channels/601",  # （cashier）
    # "rtsp://admin:SFC@dmin2022@24.109.42.130:554/Streaming/channels/701",  # （cashier）
    # "rtsp://admin:SFC@dmin2022@24.109.42.130:554/Streaming/channels/501",  # （door）
    # "rtsp://admin:SFC@dmin2022@24.109.42.130:554/Streaming/channels/1701",  # （meat）
    # "rtsp://admin:SFC@dmin2022@24.109.42.130:554/Streaming/channels/901",
    # "rtsp://admin:SFC@dmin2022@24.109.42.130:554/Streaming/channels/1201",
    # "rtsp://admin:SFC@dmin2022@24.109.42.130:554/Streaming/channels/1401"
    "rtsp://admin:hik56789@50.67.152.104:15554/Streaming/Channels/501"
]
RECONNECT_DELAY_SECONDS = 5 # Increased slightly, ffmpeg startup might take longer
SMALL_WIDTH = 1080
SMALL_HEIGHT = 720
FFPROBE_TIMEOUT = 5 # Timeout in seconds for ffprobe connection attempt

# --- Global Variables ---
frames_dict = {i: None for i in range(len(camera_urls))}
stop_event = threading.Event()
# frame_lock = threading.Lock() # Optional lock

# --- Camera Capture Thread Function with Reconnection (using ffmpeg-python) ---
def capture_camera(camera_idx, url):
    global frames_dict
    print(f"[*] Starting thread for camera {camera_idx}: {url}")

    while not stop_event.is_set():
        process = None
        width = 0
        height = 0
        frame_size = 0

        try:
            print(f"[*] Probing camera {camera_idx} for stream info...")
            # 1. Probe the stream to get video dimensions
            # Add a timeout to ffprobe to prevent indefinite hangs
            try:
                ffprobe_path = r'C:\python\ffmpeg-7.0.2-full_build\bin\ffprobe.exe'  # 定义路径变量
                probe = ffmpeg.probe(url, cmd=ffprobe_path, timeout=FFPROBE_TIMEOUT * 1000000)
            except ffmpeg.Error as e:
                print(f"[!] ffprobe error for camera {camera_idx}: {e.stderr.decode() if e.stderr else str(e)}")
                raise ValueError("ffprobe failed") # Trigger outer except block for reconnect

            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream is None:
                raise ValueError("No video stream found")

            width = int(video_stream['width'])
            height = int(video_stream['height'])
            frame_size = width * height * 3 # 3 bytes per pixel for BGR24
            print(f"[*] Camera {camera_idx}: Found video stream {width}x{height}")

            print(f"[*] Attempting to connect to camera {camera_idx} using TCP...")
            # 2. Define the ffmpeg process
            process_args = (
                ffmpeg
                .input(url,
                       rtsp_transport='tcp'
                       # 正确设置方式 (单位是微秒)
                       # **{'-stimeout': '10000000'}  # 例如设置为 10 秒
                       )
                .output('pipe:', format='rawvideo', pix_fmt='bgr24')
                .global_args('-hide_banner', '-v', 'error')
                .compile()  # Get the command line arguments
            )

            # --- === ADD THIS MISSING BLOCK === ---
            # Define the FULL path to your ffmpeg.exe
            # Replace with your actual path! Use raw string (r'') or double backslashes (\\).
            ffmpeg_path = r'C:\python\ffmpeg-7.0.2-full_build\bin\ffmpeg.exe'  # <-- Make sure this path is correct!

            # Check if the first argument in the compiled list is 'ffmpeg' and replace it
            if process_args and process_args[0].lower() == 'ffmpeg':
                print(f"[*] Replacing 'ffmpeg' command with full path: {ffmpeg_path}")  # Add for confirmation
                process_args[0] = ffmpeg_path
            # --- === END OF MISSING BLOCK === ---

            # 3. Run the ffmpeg process (Now using the full path if modification was successful)
            process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            print(f"[*] Connection process started for camera {camera_idx}.")

            # 4. Inner loop to read frames from the process stdout
            while not stop_event.is_set():
                # Read exactly one frame's worth of bytes
                in_bytes = process.stdout.read(frame_size)

                if not in_bytes:
                    # stdout pipe closed, stream ended or ffmpeg process terminated
                    print(f"[!] No more bytes from ffmpeg stdout for camera {camera_idx}. Stream likely ended or process died.")
                    # Check stderr for errors
                    stderr_data = process.stderr.read()
                    if stderr_data:
                        print(f"[!] FFmpeg stderr for camera {camera_idx}:\n{stderr_data.decode(errors='ignore')}")
                    break # Exit inner loop to trigger reconnection

                if len(in_bytes) == frame_size:
                    # Successfully read a complete frame
                    frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                    # with frame_lock:
                    frames_dict[camera_idx] = frame
                else:
                    # Should not happen if stdout isn't closed, but handle incomplete read
                    print(f"[!] Incomplete frame read for camera {camera_idx} ({len(in_bytes)}/{frame_size} bytes). Triggering reconnect...")
                    break # Exit inner loop

        except Exception as e:
            print(f"[!] Error with camera {camera_idx}: {type(e).__name__}: {e}. Attempting reconnect in {RECONNECT_DELAY_SECONDS}s...")
            # with frame_lock:
            frames_dict[camera_idx] = None # Clear frame on error

        finally:
            # Ensure ffmpeg process is terminated if it was started
            if process and process.poll() is None: # Check if process is still running
                print(f"[*] Terminating ffmpeg process for camera {camera_idx}...")
                process.terminate() # Send SIGTERM
                try:
                    process.wait(timeout=2) # Wait for termination
                    print(f"[*] FFmpeg process for camera {camera_idx} terminated gracefully.")
                    # Optionally read remaining stderr after termination
                    stderr_data = process.stderr.read()
                    if stderr_data:
                        print(f"[!] Remaining FFmpeg stderr for camera {camera_idx}:\n{stderr_data.decode(errors='ignore')}")
                except subprocess.TimeoutExpired:
                    print(f"[!] FFmpeg process for camera {camera_idx} did not terminate gracefully, killing.")
                    process.kill() # Force kill if terminate fails
                    process.wait() # Wait for kill
                # Close pipes
                if process.stdout: process.stdout.close()
                if process.stderr: process.stderr.close()

        # Wait before the next reconnection attempt, checking stop_event
        if not stop_event.is_set():
            print(f"[*] Camera {camera_idx} waiting {RECONNECT_DELAY_SECONDS}s before reconnect attempt.")
            for _ in range(RECONNECT_DELAY_SECONDS):
                if stop_event.is_set():
                    break
                time.sleep(1)

    print(f"[*] Capture thread for camera {camera_idx} stopped.")


# --- Main Program ---
# (The main program loop for display remains unchanged as it reads from frames_dict)
print("[*] Starting camera capture threads...")
threads = []
for idx, url in enumerate(camera_urls):
    # Use daemon=True so threads exit automatically if main program exits unexpectedly
    t = threading.Thread(target=capture_camera, args=(idx, url), daemon=True)
    t.start()
    threads.append(t)

# Calculate layout
num_cams = len(camera_urls)
if num_cams == 8:
    grid_cols = 4
    grid_rows = 2
elif num_cams == 6:
    grid_cols = 3
    grid_rows = 2
else:
    grid_cols = int(np.ceil(np.sqrt(num_cams)))
    grid_rows = int(np.ceil(num_cams / grid_cols))

print(f"[*] Display layout: {grid_rows} rows x {grid_cols} columns")
print("[*] Starting display loop. Press 'q' to quit.")

# Main display loop
while True:
    all_rows = []
    try:
        for r in range(grid_rows):
            row_frames = []
            for c in range(grid_cols):
                idx = r * grid_cols + c
                if idx < num_cams:
                    # with frame_lock: # Optional lock
                    frame = frames_dict.get(idx)

                    if frame is not None:
                        # Resize here to avoid doing it in the capture thread
                        small_frame = cv2.resize(frame, (SMALL_WIDTH, SMALL_HEIGHT))
                    else:
                        # Create a black frame with text indicating connection status
                        small_frame = np.zeros((SMALL_HEIGHT, SMALL_WIDTH, 3), dtype=np.uint8)
                        cv2.putText(small_frame, f"CAM {idx}", (10, SMALL_HEIGHT // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 1)
                        # Check if thread is alive for more specific status (optional)
                        status_text = "No Signal"
                        if idx < len(threads) and not threads[idx].is_alive() and not stop_event.is_set():
                            status_text = "Thread Error"
                        elif frame is None and threads[idx].is_alive():
                             status_text = "Connecting..."

                        cv2.putText(small_frame, status_text, (10, SMALL_HEIGHT // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 1)
                else:
                    # If not enough cameras to fill the grid, add empty placeholders
                    small_frame = np.zeros((SMALL_HEIGHT, SMALL_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(small_frame, "EMPTY", (10, SMALL_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (100, 100, 100), 1)

                row_frames.append(small_frame)

            # Combine frames in the current row
            if row_frames:
                current_row_image = np.hstack(row_frames)
                all_rows.append(current_row_image)

        # Combine all rows vertically
        if all_rows:
            merged_frame = np.vstack(all_rows)
            cv2.imshow('Multi-Camera Monitor', merged_frame)
        else:
            # If threads haven't started providing frames yet
             placeholder_display = np.zeros((SMALL_HEIGHT * grid_rows, SMALL_WIDTH * grid_cols, 3), dtype=np.uint8)
             cv2.putText(placeholder_display, "Initializing...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
             cv2.imshow('Multi-Camera Monitor', placeholder_display)


        # Check for quit key
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("[*] 'q' pressed. Stopping threads...")
            stop_event.set()
            break

    except KeyboardInterrupt:
        print("[*] Ctrl+C detected. Stopping threads...")
        stop_event.set()
        break
    except Exception as e:
        print(f"[!] Error in main display loop: {e}")
        # Decide whether to stop or continue
        time.sleep(1) # Prevent rapid error loops

# --- Cleanup ---
print("[*] Waiting for capture threads to join...")
for t in threads:
    # Give threads time to react to stop_event and terminate ffmpeg
    t.join(timeout=RECONNECT_DELAY_SECONDS + 3)

print("[*] Closing OpenCV windows.")
cv2.destroyAllWindows()
print("[*] Program finished.")