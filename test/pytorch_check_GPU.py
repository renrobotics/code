import torch

def check_gpu_availability():
    """检查 PyTorch 是否可以访问并使用 CUDA (NVIDIA GPU)。"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"PyTorch 可以访问 {gpu_count} 个 CUDA 支持的 GPU(s)。")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("PyTorch 无法找到任何 CUDA 支持的 GPU。请确保你的 NVIDIA 驱动已正确安装。")
        return False

if __name__ == "__main__":
    print("--- 检查 PyTorch GPU 支持 ---")
    gpu_available = check_gpu_availability()

    if gpu_available:
        # 可选：进行一个简单的 GPU 张量操作测试
        try:
            # 创建一个在 GPU 上的张量
            device = torch.device("cuda")
            a = torch.randn(3, 3).to(device)
            b = torch.randn(3, 3).to(device)
            c = torch.matmul(a, b)
            print("\n--- GPU 张量操作测试 ---")
            print("在 GPU 上创建的张量 a:\n", a)
            print("在 GPU 上创建的张量 b:\n", b)
            print("在 GPU 上进行矩阵乘法得到的张量 c:\n", c)
            print("张量 c 所在的设备:", c.device)
            print("GPU 测试成功！")
        except Exception as e:
            print(f"\n--- GPU 张量操作测试失败 ---")
            print(f"错误信息: {e}")
    else:
        print("\n请检查你的 PyTorch 安装是否包含 CUDA 支持。")