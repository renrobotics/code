import tensorrt as trt
import sys

def check_tensorrt_installation():
    print(f"Python version: {sys.version}")
    print(f"TensorRT version: {trt.__version__}") # 检查 TensorRT Python 包的版本

    # 尝试创建一个基本的 TensorRT logger 和 builder
    # 这会尝试链接到核心的 TensorRT 库 (如 nvinfer.dll)
    try:
        logger = trt.Logger(trt.Logger.INFO) # 或者 trt.Logger.WARNING
        print("TensorRT Logger created successfully.")

        builder = trt.Builder(logger)
        print("TensorRT Builder created successfully.")

        # (可选) 进一步检查是否有可用的 GPU 设备
        if trt.get_trt_versions()[0] >= 8000000: # get_trt_versions() returns tuple (trt_version, cudnn_version, cublas_version)
             # For TensorRT 8.x and later, explicit GPU device count check is different
             # We rely on Builder creation as an indicator with GPU.
             # A more direct check might involve trying to build a dummy network.
             print("TensorRT core seems accessible (Builder created).")
        else: # Older TensorRT versions might have different device checks
             print("TensorRT version might be older, specific GPU device count check might vary.")


        print("--- TensorRT Installation Check Successful ---")

    except AttributeError as ae:
        print(f"AttributeError during TensorRT check: {ae}")
        print("This might indicate an issue with the TensorRT Python bindings or an incomplete installation.")
    except Exception as e:
        print(f"An unexpected error occurred during TensorRT check: {e}")
        print("Ensure core TensorRT libraries (like nvinfer.dll) are accessible and compatible.")
        print("Check if the TensorRT 'lib' directory is in your system PATH.")

if __name__ == '__main__':
    check_tensorrt_installation()