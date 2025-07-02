#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速测试脚本
用于验证模型加载和基本功能是否正常
"""

import os
import sys
import yaml
import torch
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_builder import build_model


def test_model_loading(config_path='configs/config.yaml', checkpoint_path='checkpoints/best_model.pth'):
    """测试模型加载"""
    print("=== Model Loading Test ===")
    
    try:
        # 加载配置
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✓ Config loaded successfully")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {device}")
        
        # 构建模型
        print("Building model...")
        model = build_model(config, device)
        print("✓ Model built successfully")
        
        # 加载checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded model_state_dict from checkpoint")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("✓ Loaded state_dict from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded checkpoint directly")
        
        model.eval()
        print("✓ Model set to evaluation mode")
        
        return model, config, device
        
    except Exception as e:
        print(f"✗ Error during model loading: {str(e)}")
        return None, None, None


def test_model_inference(model, config, device):
    """测试模型推理"""
    print("\n=== Model Inference Test ===")
    
    try:
        # 创建虚拟输入数据
        transforms_config = config['transforms']
        num_frames = transforms_config['num_frames']
        crop_size = transforms_config['crop_size']
        alpha = transforms_config['alpha']
        
        print(f"Creating dummy input: {num_frames} frames, {crop_size}x{crop_size}")
        
        # 根据SlowFast模型的正确输入格式创建数据
        # 创建一个完整的帧序列 [batch, channels, temporal, height, width]
        full_frames = torch.randn(1, 3, num_frames, crop_size, crop_size).to(device)
        
        # 按照PyTorch Hub文档的PackPathway实现
        # Fast pathway: 使用所有帧
        fast_pathway = full_frames
        
        # Slow pathway: 从fast pathway中采样（每alpha帧取一帧）
        slow_indices = torch.linspace(0, num_frames - 1, num_frames // alpha).long().to(device)
        slow_pathway = torch.index_select(full_frames, 2, slow_indices)
        
        print(f"✓ Slow pathway shape: {slow_pathway.shape}")
        print(f"✓ Fast pathway shape: {fast_pathway.shape}")
        
        # 前向传播
        print("Running forward pass...")
        with torch.no_grad():
            outputs = model([slow_pathway, fast_pathway])
        
        print(f"✓ Model output shape: {outputs.shape}")
        print(f"✓ Expected classes: {config['data']['num_classes']}")
        
        # 检查输出维度
        if outputs.shape[1] == config['data']['num_classes']:
            print("✓ Output dimensions match expected number of classes")
        else:
            print(f"✗ Output dimensions mismatch: got {outputs.shape[1]}, expected {config['data']['num_classes']}")
        
        # 计算概率
        probabilities = torch.softmax(outputs, dim=1)
        probs = probabilities.cpu().numpy()[0]
        
        print(f"✓ Probabilities: {probs}")
        print(f"✓ Predicted class: {np.argmax(probs)}")
        print(f"✓ Confidence: {np.max(probs):.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during inference: {str(e)}")
        return False


def test_video_processing():
    """测试视频处理组件"""
    print("\n=== Video Processing Test ===")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        
        # 测试虚拟帧处理
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"✓ Created dummy frame: {dummy_frame.shape}")
        
        # 测试基本的OpenCV操作
        resized = cv2.resize(dummy_frame, (224, 224))
        print(f"✓ Frame resize test: {resized.shape}")
        
        # 测试颜色空间转换
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        print(f"✓ Color conversion test: {rgb_frame.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during video processing test: {str(e)}")
        return False


def main():
    print("SlowFast Theft Detection Model - Quick Test")
    print("=" * 50)
    
    # 检查文件存在性
    config_path = 'configs/config.yaml'
    checkpoint_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint file not found: {checkpoint_path}")
        return
    
    # 测试模型加载
    model, config, device = test_model_loading(config_path, checkpoint_path)
    if model is None:
        print("✗ Model loading failed, stopping tests")
        return
    
    # 测试模型推理
    inference_success = test_model_inference(model, config, device)
    if not inference_success:
        print("✗ Model inference failed")
        return
    
    # 测试视频处理
    video_success = test_video_processing()
    if not video_success:
        print("✗ Video processing test failed")
        return
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! The model is ready for use.")
    print("You can now run:")
    print("  python test_model.py --video /path/to/video.mp4")
    print("  python test_model.py --rtsp rtsp://camera.url")
    print("  python batch_test.py /path/to/video/directory")


if __name__ == '__main__':
    main() 