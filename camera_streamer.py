# camera_streamer.py
"""
Handles reading frames from a single RTSP camera stream using ffmpeg-python
and puts raw frames into an input queue. Includes reconnection.
"""
import threading
import subprocess
import time
import numpy as np
import ffmpeg
import config # Import configuration
import queue # Required for queue operations (like checking Full exception)

class CameraStreamer:
    def __init__(self, camera_idx, rtsp_url, input_queue, stop_event):
        """
        Initializes the CameraStreamer.

        Args:
            camera_idx (int): A unique index for this camera stream.
            rtsp_url (str): The RTSP URL of the camera stream.
            input_queue (queue.Queue): The queue to put successfully read frames into.
            stop_event (threading.Event): An event to signal when the thread should stop.
        """
        self.camera_idx = camera_idx
        self.rtsp_url = rtsp_url
        self.input_queue = input_queue # Queue to put (camera_idx, raw_frame) tuples
        self.stop_event = stop_event
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.is_running = False
        self.width = 0
        self.height = 0

    def _run(self):
        """The main loop for the camera reading thread."""
        print(f"[*] Starting thread for camera {self.camera_idx}: {self.rtsp_url}")
        self.is_running = True

        while not self.stop_event.is_set():
            process = None # Holds the ffmpeg subprocess
            frame_size = 0 # Calculated size of one frame in bytes

            try:
                # 1. Probe stream information (with timeout)
                # print(f"[DEBUG] Cam {self.camera_idx}: Entering probe section in loop...")
                try:
                    # print(f"[DEBUG] Cam {self.camera_idx}: ABOUT TO CALL ffmpeg.probe...")
                    # Use ffprobe to get stream details like width and height
                    probe = ffmpeg.probe(self.rtsp_url, cmd=config.FFPROBE_PATH, timeout=config.FFPROBE_TIMEOUT * 1000000)
                    # print(f"[DEBUG] Cam {self.camera_idx}: Probe call FINISHED.")
                except ffmpeg.Error as e_probe:
                    print(f"[!] Cam {self.camera_idx}: Probe failed. Error: {type(e_probe).__name__}")
                    if e_probe.stderr:
                        print(f"    FFprobe stderr: {e_probe.stderr.decode(errors='ignore')}")
                    raise ValueError("ffprobe failed or timed out") # Trigger outer except for retry

                # Find the video stream in the probe results
                video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                if video_stream is None:
                    raise ValueError("No video stream found in probe results")

                # Get dimensions
                self.width = int(video_stream.get('width', 0))
                self.height = int(video_stream.get('height', 0))
                if self.width <= 0 or self.height <= 0:
                    raise ValueError(f"Invalid dimensions from probe: {self.width}x{self.height}")

                # Calculate frame size (assuming BGR format, 3 bytes per pixel)
                frame_size = self.width * self.height * 3
                print(f"[*] Cam {self.camera_idx}: Probe successful. Dimensions: {self.width}x{self.height}")

                # 2. Prepare and Start the main ffmpeg process
                # Define arguments for ffmpeg using ffmpeg-python
                process_args = (
                    ffmpeg
                    # Input options: RTSP URL and force TCP transport (often more reliable than UDP for RTSP)
                    # Removed the problematic 'stimeout' argument
                    .input(self.rtsp_url, rtsp_transport='tcp')
                    # Output options: pipe output, raw video format, BGR pixel format (for OpenCV), set output frame rate
                    .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=config.FFMPEG_FPS_OUTPUT)
                    # Global options: hide banner, set log level to error to reduce noise
                    .global_args('-hide_banner', '-v', 'error')
                    # Compile arguments into a list for subprocess
                    .compile()
                )

                # Ensure the command uses the full path to ffmpeg specified in config
                if process_args and process_args[0].lower() == 'ffmpeg':
                    process_args[0] = config.FFMPEG_PATH

                print(f"[DEBUG] Cam {self.camera_idx}: ABOUT TO CALL subprocess.Popen with args: {process_args}")
                # Start the ffmpeg process
                process = subprocess.Popen(
                    process_args,
                    stdout=subprocess.PIPE, # Capture standard output (video data)
                    stderr=subprocess.PIPE, # Capture standard error (error messages)
                    bufsize=frame_size * 2 # Set a reasonable buffer size (e.g., for a couple of frames)
                )
                print(f"[DEBUG] Cam {self.camera_idx}: subprocess.Popen call FINISHED. Process PID: {process.pid if process else 'None'}")

                # 3. Read frames continuously from the ffmpeg process stdout pipe
                while not self.stop_event.is_set():
                    # print(f"[DEBUG] Cam {self.camera_idx}: ABOUT TO CALL process.stdout.read({frame_size})...")
                    # Read exactly one frame's worth of bytes
                    in_bytes = process.stdout.read(frame_size)
                    bytes_read = len(in_bytes)
                    # print(f"[DEBUG] Cam {self.camera_idx}: process.stdout.read returned {bytes_read} bytes.")

                    if not in_bytes:
                        # If read returns empty bytes, the pipe was likely closed (ffmpeg exited)
                        print(f"[!] Cam {self.camera_idx}: No bytes read from ffmpeg stdout pipe. Stream ended or process died.")
                        # Check stderr immediately for clues
                        if process:
                             stderr_data = process.stderr.read()
                             if stderr_data:
                                 print(f"[!] Cam {self.camera_idx}: FFmpeg stderr after read failure:\n{stderr_data.decode(errors='ignore')}")
                        break # Exit inner loop to trigger cleanup and reconnect attempt

                    if bytes_read == frame_size:
                        # Successfully read a complete frame
                        frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
                        # print(f"[DEBUG] Cam {self.camera_idx}: Frame decoded. Putting into queue...") # Reduce verbosity
                        try:
                            # Put the frame and its index into the input queue for processing
                            self.input_queue.put((self.camera_idx, frame), block=True, timeout=2) # Block with timeout
                        except queue.Full:
                            # If the queue is full (processing falling behind), drop the frame
                             print(f"[!] Cam {self.camera_idx}: Input queue full. Dropping frame.")
                             # Consider adding a small sleep here if this happens frequently
                             time.sleep(0.05)
                             continue
                    else:
                        # Read an unexpected number of bytes (error condition)
                        print(f"[!] Cam {self.camera_idx}: Incomplete frame read ({bytes_read}/{frame_size}). Triggering reconnect.")
                        break # Exit inner loop

            except Exception as e:
                # Catch any exception during probe or ffmpeg process handling
                print(f"[!] Error in Cam {self.camera_idx} thread: {type(e).__name__}: {e}. Retrying...")
                # Print traceback for detailed debugging if needed
                # import traceback
                # traceback.print_exc()
                # Ensure frame is None in the main dict if this thread has an error
                # (Not strictly necessary here as it's cleared in finally/next loop iteration)

            finally:
                # --- Cleanup ffmpeg process ---
                # This block runs whether the try block succeeded or failed
                if process and process.poll() is None: # Check if process was started and is still running
                    print(f"[DEBUG] Cam {self.camera_idx}: Cleaning up ffmpeg process {process.pid}...")
                    # Read any remaining stderr messages first
                    stderr_data = process.stderr.read()
                    if stderr_data:
                        print(f"[DEBUG] Cam {self.camera_idx}: FFmpeg stderr from finally block:\n{stderr_data.decode(errors='ignore')}")

                    # Terminate the process gracefully
                    process.terminate()
                    try:
                        process.wait(timeout=1) # Wait briefly for termination
                    except subprocess.TimeoutExpired:
                        # Force kill if terminate doesn't work quickly
                        print(f"[!] Cam {self.camera_idx}: FFmpeg process did not terminate gracefully, killing.")
                        process.kill()
                        process.wait() # Wait for kill to complete
                # Ensure pipes are closed
                if process and process.stdout: process.stdout.close()
                if process and process.stderr: process.stderr.close()
                # Attempt to release the process object reference
                del process

            # --- Reconnect Delay ---
            # Wait before the next connection attempt in the outer loop,
            # but only if the stop event hasn't been set.
            if not self.stop_event.is_set():
                # print(f"[*] Cam {self.camera_idx} waiting {config.RECONNECT_DELAY_SECONDS}s before reconnect attempt.") # Less verbose
                # Check stop event frequently during sleep
                for _ in range(config.RECONNECT_DELAY_SECONDS):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)

        # Loop exited because stop_event was set
        self.is_running = False
        print(f"[*] Capture thread for camera {self.camera_idx} stopped.")

    def start(self):
        """Starts the camera reading thread."""
        if not self.is_running:
            self.thread.start()

    def is_alive(self):
        """Checks if the reading thread is currently running."""
        return self.thread.is_alive()