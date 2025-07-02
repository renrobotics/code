#!/usr/bin/env python3
"""
测试RTSP流的稳定可视化效果
"""

import cv2
import numpy as np
import argparse
import os
import time
from test_model import TheftDetector

def test_stable_rtsp_visualization():
    """测试稳定的RTSP可视化"""
    
    # 检查必要文件
    config_path = 'configs/config.yaml'
    checkpoint_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    print("Loading theft detection model...")
    detector = TheftDetector(config_path, checkpoint_path)
    
    # 使用本地视频文件模拟RTSP流
    test_video = '../videos/2.mp4'  # 或者使用真实的RTSP URL
    
    if not os.path.exists(test_video):
        print(f"Test video not found: {test_video}")
        print("Please provide a real RTSP URL using --rtsp parameter")
        return
    
    print(f"Testing stable visualization with: {test_video}")
    print("Press 'q' to quit")
    print("Note: This simulates RTSP stream behavior with stable visualization")
    
    # 模拟RTSP流测试
    detector.test_rtsp_stream(test_video, duration=30)  # 测试30秒

def main():
    parser = argparse.ArgumentParser(description='Test stable RTSP visualization')
    parser.add_argument('--rtsp', type=str, help='RTSP stream URL')
    parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
    
    args = parser.parse_args()
    
    if args.rtsp:
        # 使用真实RTSP流
        config_path = 'configs/config.yaml'
        checkpoint_path = 'checkpoints/best_model.pth'
        
        print("Loading theft detection model...")
        detector = TheftDetector(config_path, checkpoint_path)
        
        print(f"Testing RTSP stream: {args.rtsp}")
        print("Press 'q' to quit")
        
        detector.test_rtsp_stream(args.rtsp, args.duration)
    else:
        # 使用模拟测试
        test_stable_rtsp_visualization()

if __name__ == '__main__':
    main() 