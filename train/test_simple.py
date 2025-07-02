#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单测试脚本
用于快速验证可视化功能是否正常工作
"""

import os
import sys
import cv2
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from test_model import TheftDetector


def create_test_video(output_path, duration=5, fps=25):
    """创建一个简单的测试视频"""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        # 创建一个简单的测试帧
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加一些图案
        cv2.rectangle(frame, (50, 50), (width-50, height-50), (50, 50, 50), -1)
        cv2.circle(frame, (width//2, height//2), 80, (100, 100, 100), -1)
        
        # 添加帧数显示
        text = f"Frame: {i+1}/{total_frames}"
        cv2.putText(frame, text, (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")


def main():
    # 创建测试视频
    test_video_path = "test_input.mp4"
    output_video_path = "test_output.mp4"
    
    print("Creating test video...")
    create_test_video(test_video_path, duration=3, fps=25)
    
    print("Testing theft detection...")
    try:
        # 创建检测器
        detector = TheftDetector('configs/config.yaml', 'checkpoints/best_model.pth')
        
        # 测试视频
        detector.test_video_file(test_video_path, output_video_path)
        
        print(f"Test completed! Check output video: {output_video_path}")
        
        # 清理测试文件
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
            print(f"Cleaned up test input: {test_video_path}")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        # 清理测试文件
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


if __name__ == '__main__':
    main() 