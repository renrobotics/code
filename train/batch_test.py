#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量测试脚本
用于批量测试多个视频文件
"""

import os
import sys
import yaml
import glob
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from test_model import TheftDetector


def batch_test_videos(video_dir: str, config_path: str = 'configs/config.yaml', 
                     checkpoint_path: str = 'checkpoints/best_model.pth',
                     output_dir: str = None, video_extensions: list = None):
    """
    批量测试视频文件
    
    Args:
        video_dir: 视频文件目录
        config_path: 配置文件路径
        checkpoint_path: 模型checkpoint路径
        output_dir: 输出目录（可选）
        video_extensions: 支持的视频格式
    """
    if video_extensions is None:
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    
    # 创建检测器
    print("Initializing theft detector...")
    detector = TheftDetector(config_path, checkpoint_path)
    
    # 查找所有视频文件
    video_files = []
    for ext in video_extensions:
        pattern = os.path.join(video_dir, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    results = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        # 设置输出路径
        output_path = None
        if output_dir:
            video_name = Path(video_path).stem
            output_path = os.path.join(output_dir, f"{video_name}_detected.mp4")
        
        try:
            # 测试视频
            detector.test_video_file(video_path, output_path)
            results.append({
                'video': video_path,
                'status': 'success',
                'output': output_path
            })
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            results.append({
                'video': video_path,
                'status': 'error',
                'error': str(e)
            })
    
    # 输出总结
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    print(f"Total videos processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    
    if error_count > 0:
        print("\nFailed videos:")
        for result in results:
            if result['status'] == 'error':
                print(f"  - {result['video']}: {result['error']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch test theft detection model')
    parser.add_argument('video_dir', type=str, help='Directory containing video files')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, help='Output directory for processed videos')
    parser.add_argument('--extensions', nargs='+', 
                        default=['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv'],
                        help='Video file extensions to process')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        return
    
    batch_test_videos(
        video_dir=args.video_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        video_extensions=args.extensions
    )


if __name__ == '__main__':
    main() 