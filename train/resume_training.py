#!/usr/bin/env python3
"""
Resume Training Helper Script
用于管理训练断点续训的辅助脚本
"""

import os
import argparse
import torch
from datetime import datetime


def list_checkpoints(checkpoint_dir: str = "checkpoints") -> None:
    """列出所有可用的checkpoint文件"""
    print("\n=== Available Checkpoints ===")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' not found!")
        return
    
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            checkpoint_files.append((file, file_size, mod_time))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x[2], reverse=True)
    
    for i, (filename, size, mod_time) in enumerate(checkpoint_files, 1):
        print(f"{i}. {filename}")
        print(f"   Size: {size:.1f} MB")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Try to load and show checkpoint info
        try:
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch'] + 1}")
            if 'best_val_accuracy' in checkpoint:
                print(f"   Best Accuracy: {checkpoint['best_val_accuracy']:.4f}")
            if 'val_accuracy' in checkpoint:
                print(f"   Last Accuracy: {checkpoint['val_accuracy']:.4f}")
        except Exception as e:
            print(f"   (Could not read checkpoint info: {e})")
        print()


def check_checkpoint_info(checkpoint_path: str) -> None:
    """检查特定checkpoint的详细信息"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file '{checkpoint_path}' not found!")
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"\n=== Checkpoint Info: {checkpoint_path} ===")
        
        for key, value in checkpoint.items():
            if key.endswith('_state_dict'):
                print(f"{key}: <state_dict with {len(value)} keys>")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Error loading checkpoint: {e}")


def backup_checkpoint(checkpoint_path: str) -> None:
    """备份checkpoint文件"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file '{checkpoint_path}' not found!")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{checkpoint_path}.backup_{timestamp}"
    
    try:
        import shutil
        shutil.copy2(checkpoint_path, backup_path)
        print(f"Checkpoint backed up to: {backup_path}")
    except Exception as e:
        print(f"Error backing up checkpoint: {e}")


def clean_old_checkpoints(checkpoint_dir: str = "checkpoints", keep_count: int = 5) -> None:
    """清理旧的checkpoint文件，只保留最新的几个"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' not found!")
        return
    
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth') and 'backup' not in file and file != 'best_model.pth':
            file_path = os.path.join(checkpoint_dir, file)
            mod_time = os.path.getmtime(file_path)
            checkpoint_files.append((file, file_path, mod_time))
    
    if len(checkpoint_files) <= keep_count:
        print(f"Only {len(checkpoint_files)} checkpoint files found, no cleanup needed.")
        return
    
    # Sort by modification time (oldest first)
    checkpoint_files.sort(key=lambda x: x[2])
    
    files_to_delete = checkpoint_files[:-keep_count]
    
    print(f"\nCleaning up {len(files_to_delete)} old checkpoint files:")
    for filename, file_path, _ in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {filename}")
        except Exception as e:
            print(f"Error deleting {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Resume Training Helper")
    parser.add_argument("--list", action="store_true", help="List all available checkpoints")
    parser.add_argument("--info", type=str, help="Show detailed info for a specific checkpoint")
    parser.add_argument("--backup", type=str, help="Backup a specific checkpoint")
    parser.add_argument("--clean", action="store_true", help="Clean old checkpoint files")
    parser.add_argument("--keep", type=int, default=5, help="Number of checkpoints to keep when cleaning")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints(args.checkpoint_dir)
    elif args.info:
        check_checkpoint_info(args.info)
    elif args.backup:
        backup_checkpoint(args.backup)
    elif args.clean:
        clean_old_checkpoints(args.checkpoint_dir, args.keep)
    else:
        print("Usage examples:")
        print("  python resume_training.py --list                    # 列出所有checkpoint")
        print("  python resume_training.py --info checkpoints/latest_checkpoint.pth  # 查看checkpoint信息")
        print("  python resume_training.py --backup checkpoints/latest_checkpoint.pth  # 备份checkpoint")
        print("  python resume_training.py --clean                   # 清理旧checkpoint")
        print("\nTo resume training, simply run:")
        print("  python train.py --config configs/config.yaml")
        print("  (The script will automatically detect and load the latest checkpoint)")


if __name__ == "__main__":
    main() 