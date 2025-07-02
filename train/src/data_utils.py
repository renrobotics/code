# src/data_utils.py

import os
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


def normalize_to_float(x):
    """Convert video tensor from [0, 255] to [0, 1] range."""
    return x / 255.0


class PackPathway:
    """
    自定义的 PackPathway 实现，用于为 SlowFast 模型准备双路径输入。
    
    将单个视频张量分割成 Slow 和 Fast 两个路径：
    - Slow pathway: 低帧率，捕捉语义信息
    - Fast pathway: 高帧率，捕捉快速动作
    """
    
    def __init__(self, alpha: int = 4):
        """
        Args:
            alpha (int): Fast pathway 相对于 Slow pathway 的速度倍率
        """
        self.alpha = alpha
    
    def __call__(self, frames: torch.Tensor) -> list:
        """
        Args:
            frames (torch.Tensor): 输入视频张量，形状为 (C, T, H, W)
        
        Returns:
            list: [slow_pathway, fast_pathway] 两个张量的列表
        """
        # Slow pathway: 每隔 alpha 帧取一帧
        slow_pathway = frames[:, ::self.alpha, :, :]
        
        # Fast pathway: 每帧都取，但通道数减少到 1/8
        fast_pathway = frames[:, :, :, :]
        # 为了减少计算量，Fast pathway 通常只保留部分通道
        # 这里我们简单地保留所有通道，在实际应用中可以进一步优化
        
        return [slow_pathway, fast_pathway]


def create_dataloaders(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns the training and validation dataloaders for SlowFast.

    Args:
        config (Dict[str, Any]): A dictionary containing the configuration
                                 parameters for data loading and transformations.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training
                                       and validation dataloaders.
    """
    transform_config = config["transforms"]
    slowfast_alpha = transform_config["alpha"]

    # -- Define the transformation pipelines --

    # Base transformation for both train and validation
    base_transform = Compose(
        [
            UniformTemporalSubsample(transform_config["num_frames"]),
            Lambda(normalize_to_float),  # Use named function instead of lambda
            Normalize(transform_config["mean"], transform_config["std"]),
        ]
    )

    # Training pipeline includes spatial data augmentation
    train_spatial_transform = Compose(
        [
            RandomShortSideScale(min_size=256, max_size=320),
            RandomCrop(transform_config["crop_size"]),
            RandomHorizontalFlip(p=0.5),
        ]
    )

    # Validation pipeline uses fixed scaling and center cropping
    val_spatial_transform = Compose(
        [
            ShortSideScale(size=transform_config["side_size"]),
            CenterCrop(transform_config["crop_size"]),
        ]
    )

    # Combine base transforms with spatial transforms and our custom PackPathway
    train_transform = Compose(
        [
            base_transform,
            train_spatial_transform,
            PackPathway(alpha=slowfast_alpha)
        ]
    )
    val_transform = Compose(
        [
            base_transform,
            val_spatial_transform,
            PackPathway(alpha=slowfast_alpha)
        ]
    )

    # Wrap transforms in ApplyTransformToKey
    train_transform_wrapped = ApplyTransformToKey(key="video", transform=train_transform)
    val_transform_wrapped = ApplyTransformToKey(key="video", transform=val_transform)

    # -- Create the LabeledVideoDataset for train and validation sets --
    clip_duration = (transform_config["num_frames"] * transform_config["sampling_rate"]) / 30.0
    clip_sampler = UniformClipSampler(clip_duration=clip_duration)

    # Read CSV files and create proper format for LabeledVideoDataset
    train_df = pd.read_csv(config["data"]["train_csv_path"], header=None, names=['path', 'label'])
    val_df = pd.read_csv(config["data"]["val_csv_path"], header=None, names=['path', 'label'])
    
    # Prepend video root path to each video path and create tuples
    video_root = config["data"]["video_root_path"]
    train_data_with_full_paths = [(os.path.join(video_root, row['path']), {'label': row['label']}) for _, row in train_df.iterrows()]
    val_data_with_full_paths = [(os.path.join(video_root, row['path']), {'label': row['label']}) for _, row in val_df.iterrows()]

    train_dataset = LabeledVideoDataset(
        labeled_video_paths=train_data_with_full_paths,
        clip_sampler=clip_sampler,
        transform=train_transform_wrapped,
        decode_audio=False,
    )

    val_dataset = LabeledVideoDataset(
        labeled_video_paths=val_data_with_full_paths,
        clip_sampler=clip_sampler,
        transform=val_transform_wrapped,
        decode_audio=False,
    )

    print("Train dataset created successfully.")
    print("Validation dataset created successfully.")

    # -- Create the DataLoaders --
    # Use fewer workers on Windows to avoid multiprocessing issues
    num_workers = 0 if os.name == 'nt' else config["training"]["num_workers"]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=num_workers,
        pin_memory=True,
        # shuffle=True,  # IterableDataset doesn't support shuffle in DataLoader
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, val_loader 