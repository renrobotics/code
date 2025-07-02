from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11s-pose.pt")

# Export the model to TensorRT format
model.export(format="engine",dynamic=True,batch=16)  # creates 'yolo11n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("yolo11s-pose.engine")