# src/model_builder.py

from typing import Any, Dict

import torch
from torch import nn

# 直接从pytorchvideo库导入模型，而不是通过torch.hub
from pytorchvideo.models import slowfast


def build_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Builds and configures the SlowFast model for training.

    This function loads a pre-trained SlowFast model from PyTorch Hub,
    modifies its final classification layer to match the number of classes
    in the custom dataset, and moves the model to the specified compute device.

    Args:
        config (Dict[str, Any]): A dictionary containing model and data
                                 configuration parameters.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to
                               which the model should be moved.

    Returns:
        nn.Module: The configured PyTorch model ready for training.
    """
    # 直接调用函数创建模型
    model = slowfast.create_slowfast(
        model_num_class=config["data"]["num_classes"]  # 直接设置最终的分类数
    )
    
    # 手动加载预训练权重（如果需要）
    if config["model"]["pretrained"]:
        print("Loading pre-trained weights...")
        # 加载在Kinetics上预训练好的模型
        state_dict = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R50.pyth",
            progress=True
        )
        
        # 打印所有可用的键以便调试
        print("Available keys in state_dict:")
        for key in sorted(state_dict.keys()):
            if 'proj' in key or 'head' in key or 'fc' in key or 'classifier' in key:
                print(f"  {key}: {state_dict[key].shape}")
        
        # 找到并移除分类头相关的权重
        keys_to_remove = []
        for key in state_dict.keys():
            # 查找可能的分类头键名
            if any(pattern in key for pattern in ['proj.weight', 'proj.bias', 'head.weight', 'head.bias', 'fc.weight', 'fc.bias']):
                keys_to_remove.append(key)
        
        print(f"Removing classification head keys: {keys_to_remove}")
        for key in keys_to_remove:
            state_dict.pop(key)
        
        # 加载权重，strict=False允许我们加载除了分类头之外的所有层
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        print("Pre-trained weights loaded successfully.")

    print(f"Model: {config['model']['name']} built.")
    print(f"Classifier head configured to output {config['data']['num_classes']} classes.")

    # Move the model to the specified device
    model = model.to(device)
    print(f"Model moved to device: {device}")

    return model
