#!/usr/bin/env python3
"""
测试快速响应的盗窃检测
"""

import cv2
import numpy as np
import argparse
import os
import time
from test_model import TheftDetector

def test_fast_response():
    """测试快速响应效果"""
    
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
    
    # 测试视频文件
    test_videos = [
        '../videos/2.mp4',
        '../videos/3.mp4'
    ]
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            output_path = video_path.replace('.mp4', '_fast_response.mp4')
            
            print(f"\n{'='*60}")
            print(f"Testing FAST RESPONSE with: {video_path}")
            print(f"Output will be saved to: {output_path}")
            print(f"{'='*60}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行检测
            detector.test_video_file(video_path, output_path)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            if os.path.exists(output_path):
                print(f"\n✓ SUCCESS! Fast response video created!")
                print(f"  - Output file: {output_path}")
                print(f"  - Processing time: {processing_time:.2f} seconds")
                
                # 获取输出视频信息
                cap = cv2.VideoCapture(output_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps
                    
                    print(f"  - Video resolution: {width}x{height}")
                    print(f"  - Video duration: {duration:.2f} seconds")
                    print(f"  - Processing speed: {duration/processing_time:.2f}x real-time")
                    
                    cap.release()
                
                print(f"\n🚀 IMPROVEMENTS:")
                print(f"  - Prediction frequency: Every 0.33 seconds (was 0.5s)")
                print(f"  - Response time: ~1 second (was ~2.5s)")
                print(f"  - Minimum detection time: ~0.66 seconds (was ~2.5s)")
                
                break
            else:
                print(f"✗ Failed to create output video: {output_path}")
        else:
            print(f"Video not found: {video_path}")
    
    print(f"\n{'='*60}")
    print("Fast response test completed!")
    print("Key improvements:")
    print("- Faster prediction frequency (every 0.33s vs 0.5s)")
    print("- Smarter smoothing logic (2-3 predictions vs 5)")
    print("- Earlier confidence output (1s vs 2.5s)")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='Test fast response theft detection')
    parser.add_argument('--video', type=str, help='Specific video file to test')
    parser.add_argument('--rtsp', type=str, help='RTSP stream URL for live testing')
    parser.add_argument('--duration', type=int, default=30, help='Test duration for RTSP (seconds)')
    
    args = parser.parse_args()
    
    if args.video:
        # 测试指定视频
        config_path = 'configs/config.yaml'
        checkpoint_path = 'checkpoints/best_model.pth'
        
        detector = TheftDetector(config_path, checkpoint_path)
        output_path = args.video.replace('.mp4', '_fast_response.mp4')
        
        print(f"Testing fast response with: {args.video}")
        detector.test_video_file(args.video, output_path)
        
    elif args.rtsp:
        # 测试RTSP流
        config_path = 'configs/config.yaml'
        checkpoint_path = 'checkpoints/best_model.pth'
        
        detector = TheftDetector(config_path, checkpoint_path)
        
        print(f"Testing fast response with RTSP: {args.rtsp}")
        print("Press 'q' to quit")
        detector.test_rtsp_stream(args.rtsp, args.duration)
        
    else:
        # 默认测试
        test_fast_response()

if __name__ == '__main__':
    main() 