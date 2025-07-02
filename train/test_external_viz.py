#!/usr/bin/env python3
"""
测试外部可视化面板的效果
"""

import cv2
import numpy as np
import argparse
import os
from typing import Dict, Any, List, Tuple
import yaml
import torch
import time
from collections import deque

# 导入现有的类
from test_model import TheftDetector, VideoProcessor

def test_external_visualization():
    """测试外部可视化效果"""
    
    # 检查必要文件
    config_path = 'configs/config.yaml'
    checkpoint_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    # 创建检测器
    print("Loading theft detection model...")
    detector = TheftDetector(config_path, checkpoint_path)
    
    # 测试视频文件
    test_videos = [
        '../videos/2.mp4',
        '../videos/3.mp4'
    ]
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            output_path = video_path.replace('.mp4', '_external_viz.mp4')
            print(f"\nTesting external visualization with: {video_path}")
            print(f"Output will be saved to: {output_path}")
            
            detector.test_video_file(video_path, output_path)
            
            if os.path.exists(output_path):
                print(f"✓ Success! External visualization video created: {output_path}")
                
                # 获取输出视频信息
                cap = cv2.VideoCapture(output_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    print(f"  - Resolution: {width}x{height} (extended from original)")
                    print(f"  - FPS: {fps}")
                    print(f"  - Total frames: {frame_count}")
                    
                    cap.release()
                
                break
            else:
                print(f"✗ Failed to create output video: {output_path}")
        else:
            print(f"Video not found: {video_path}")
    
    print("\nExternal visualization test completed!")

if __name__ == '__main__':
    test_external_visualization() 