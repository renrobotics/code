#!/usr/bin/env python3
"""
Convert Best Model to Checkpoint
将现有的 best_model.pth 转换为可用于断点续训的checkpoint
"""

import os
import torch
import yaml
import argparse
from src.model_builder import build_model


def convert_best_model_to_checkpoint(
    best_model_path: str,
    config_path: str,
    output_checkpoint_path: str,
    start_epoch: int = 0,
    best_accuracy: float = 0.0
) -> None:
    """将best_model.pth转换为完整的checkpoint"""
    
    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = build_model(config, device)
    
    # 加载最佳模型权重
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"Best model file {best_model_path} not found!")
        return
    
    # 创建优化器
    optimizer_name = config["training"]["optimizer"].lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config["training"]["learning_rate"], 
            weight_decay=config["training"]["weight_decay"]
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config["training"]["learning_rate"], 
            momentum=0.9, 
            weight_decay=config["training"]["weight_decay"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # 创建完整的checkpoint
    checkpoint = {
        'epoch': start_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_accuracy,
        'train_loss': 0.0,  # 未知，设为0
        'val_loss': 0.0,    # 未知，设为0
        'val_accuracy': best_accuracy,
    }
    
    # 保存checkpoint
    os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)
    torch.save(checkpoint, output_checkpoint_path)
    print(f"Checkpoint saved to {output_checkpoint_path}")
    print(f"Starting epoch: {start_epoch}")
    print(f"Best accuracy: {best_accuracy}")


def main():
    parser = argparse.ArgumentParser(description="Convert best model to checkpoint")
    parser.add_argument("--best-model", type=str, default="checkpoints/best_model.pth", 
                       help="Path to best model file")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="Path to config file")
    parser.add_argument("--output", type=str, default="checkpoints/latest_checkpoint.pth", 
                       help="Output checkpoint path")
    parser.add_argument("--start-epoch", type=int, default=0, 
                       help="Starting epoch number")
    parser.add_argument("--best-accuracy", type=float, default=0.85, 
                       help="Best validation accuracy achieved")
    
    args = parser.parse_args()
    
    convert_best_model_to_checkpoint(
        args.best_model,
        args.config,
        args.output,
        args.start_epoch,
        args.best_accuracy
    )


if __name__ == "__main__":
    main() 