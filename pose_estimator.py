# pose_estimator.py
"""
Handles loading the YOLO TensorRT engine and performing inference.
"""
import torch
from ultralytics import YOLO # Keep this import
import config # Import configuration

class PoseEstimator:
    def __init__(self, model_path=config.YOLO_MODEL_PATH):
        self.model_path = model_path
        # For TensorRT engines, device selection is often handled during engine creation
        # or by Ultralytics when loading the engine.
        # We'll let Ultralytics YOLO class handle device placement with .engine files.
        # The device used will typically be the one the engine was built for (GPU).
        self.model = self._load_model()
        # You can explicitly check device if needed after loading:
        if self.model and hasattr(self.model, 'device'):
             print(f"[PoseEstimator] Model loaded on device: {self.model.device}")


    def _load_model(self):
        print(f"[PoseEstimator] Loading TensorRT engine: {self.model_path}...")
        try:
            # Ultralytics YOLO class can load .engine files directly.
            # It should infer the task (pose) from the engine.
            # If it has trouble, you might need to specify task='pose',
            # e.g., model = YOLO(self.model_path, task='pose')
            model = YOLO(self.model_path)
            print("[PoseEstimator] TensorRT engine loaded successfully.")
            return model
        except Exception as e:
            print(f"[PoseEstimator] Error loading TensorRT engine: {e}")
            print("Ensure the .engine file exists, was built for your current GPU and TensorRT/CUDA versions,")
            print("and you have the latest Ultralytics library.")
            raise # Re-raise the exception to stop execution if model fails

    def predict(self, frame):
        """
        Runs pose estimation on a single frame using the TensorRT engine.

        Args:
            frame: The input image frame (NumPy array BGR).

        Returns:
            Ultralytics results object, or None if inference fails.
        """
        if self.model is None:
            print("[PoseEstimator] Error: Model not loaded.")
            return None
        try:
            # The .predict() API should remain largely the same.
            # Device argument might not be needed if engine is already on GPU.
            results = self.model.predict(frame,
                                         classes=[config.TARGET_CLASS_ID],
                                         conf=config.CONFIDENCE_THRESHOLD,
                                         # device=self.device, # Often not needed for .engine
                                         verbose=False,
                                         batch=64)
            return results[0] # Return the results object for the single frame
        except Exception as e:
            print(f"[PoseEstimator] Error during TensorRT prediction: {e}")
            return None

    def plot(self, results_obj, frame):
        """
        Uses the ultralytics plot() method to draw results on the frame.
        This method should still work as it operates on the results object.

        Args:
            results_obj: The Ultralytics results object from predict().
            frame: The original frame to draw on (or the resized frame if pre-resized).

        Returns:
            The annotated frame (NumPy array BGR).
        """
        if results_obj and hasattr(results_obj, 'keypoints') and results_obj.keypoints is not None:
             try:
                 return results_obj.plot(
                     conf=False, # Confidence already filtered during predict
                     boxes=config.PLOT_BOXES,
                     labels=config.PLOT_LABELS, # Keep as is, will show class name
                     line_width=config.PLOT_LINE_WIDTH,
                     font_size=config.PLOT_FONT_SIZE,
                     font=config.PLOT_FONT,
                     pil=config.PLOT_PIL,
                     img=frame
                 )
             except Exception as e:
                  print(f"[PoseEstimator] Error during plotting: {e}")
                  return frame.copy() # Return a copy of the original frame on plotting error
        else:
            return frame.copy() # Return a copy of original frame if no results or keypoints