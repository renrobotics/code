#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型测试脚本
支持测试本地视频文件和RTSP流
"""

import os
import sys
import argparse
import yaml
import cv2
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import time
from collections import deque

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_builder import build_model


class VideoProcessor:
    """视频处理器，用于预处理视频帧"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transforms_config = config['transforms']
        
        # 视频参数
        self.side_size = self.transforms_config['side_size']
        self.crop_size = self.transforms_config['crop_size']
        self.num_frames = self.transforms_config['num_frames']
        self.sampling_rate = self.transforms_config['sampling_rate']
        self.alpha = self.transforms_config['alpha']
        
        # 归一化参数
        self.mean = np.array(self.transforms_config['mean'])
        self.std = np.array(self.transforms_config['std'])
        
        # 帧缓冲区
        self.frame_buffer = deque(maxlen=self.num_frames * self.sampling_rate)
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理单个帧"""
        # 调整大小
        h, w = frame.shape[:2]
        if h < w:
            new_h, new_w = self.side_size, int(w * self.side_size / h)
        else:
            new_h, new_w = int(h * self.side_size / w), self.side_size
            
        frame = cv2.resize(frame, (new_w, new_h))
        
        # 中心裁剪
        start_h = (new_h - self.crop_size) // 2
        start_w = (new_w - self.crop_size) // 2
        frame = frame[start_h:start_h + self.crop_size, start_w:start_w + self.crop_size]
        
        # 转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # 标准化
        frame = (frame - self.mean) / self.std
        
        return frame
    
    def add_frame(self, frame: np.ndarray):
        """添加帧到缓冲区"""
        processed_frame = self.preprocess_frame(frame)
        self.frame_buffer.append(processed_frame)
    
    def get_clip(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取SlowFast输入的clip"""
        if len(self.frame_buffer) < self.num_frames * self.sampling_rate:
            return None, None
        
        # 采样帧
        frames = []
        for i in range(0, self.num_frames * self.sampling_rate, self.sampling_rate):
            if i < len(self.frame_buffer):
                frames.append(self.frame_buffer[i])
        
        if len(frames) < self.num_frames:
            # 如果帧数不够，重复最后一帧
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        
        # 转换为tensor: [T, H, W, C] -> [C, T, H, W]
        frames = np.stack(frames, axis=0)  # [T, H, W, C]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()  # [C, T, H, W]，确保是float类型
        
        # 添加batch维度 [1, C, T, H, W]
        full_frames = frames.unsqueeze(0)
        
        # 创建SlowFast的双pathway输入（按照PyTorch Hub文档的PackPathway实现）
        # Fast pathway: 使用所有帧
        fast_pathway = full_frames
        
        # Slow pathway: 从fast pathway中采样（每alpha帧取一帧）
        slow_indices = torch.linspace(0, self.num_frames - 1, self.num_frames // self.alpha).long()
        slow_pathway = torch.index_select(full_frames, 2, slow_indices)
        
        return [slow_pathway, fast_pathway]


class TheftDetector:
    """盗窃行为检测器"""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 构建模型
        self.model = build_model(self.config, self.device)
        
        # 加载checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("Model loaded successfully!")
        
        # 初始化视频处理器
        self.video_processor = VideoProcessor(self.config)
        
        # 类别标签
        self.class_names = ['Normal', 'Theft']
        
        # 预测历史（用于平滑预测结果）
        self.prediction_history = deque(maxlen=10)
    
    def predict(self, clip: List[torch.Tensor]) -> Tuple[int, float, List[float]]:
        """预测clip的类别"""
        if clip[0] is None or clip[1] is None:
            return 0, 0.0, [1.0, 0.0]
        
        with torch.no_grad():
            # 移动到设备
            clip = [pathway.to(self.device) for pathway in clip]
            
            # 前向传播
            outputs = self.model(clip)
            
            # 获取概率
            probabilities = torch.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]
            
            # 获取预测类别
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]
            
            return predicted_class, confidence, probs.tolist()
    
    def smooth_prediction(self, pred_class: int, confidence: float) -> Tuple[int, float]:
        """平滑预测结果 - 优化为更快响应"""
        self.prediction_history.append((pred_class, confidence))
        
        # 如果历史记录少于2次，直接返回当前预测（但降低置信度以保守）
        if len(self.prediction_history) < 2:
            return pred_class, confidence * 0.8  # 降低置信度表示不确定
        
        # 获取最近的预测（最多3次，约1.5秒）
        recent_preds = list(self.prediction_history)[-3:]
        
        # 分析最近的预测
        theft_preds = [(p, c) for p, c in recent_preds if p == 1]
        normal_preds = [(p, c) for p, c in recent_preds if p == 0]
        
        # 计算盗窃预测的数量和平均置信度
        theft_count = len(theft_preds)
        high_conf_theft_count = sum(1 for p, c in theft_preds if c > 0.7)
        
        # 更灵敏的判定逻辑
        if len(recent_preds) >= 2:
            # 情况1: 最近2-3次中有2次以上是高置信度盗窃
            if high_conf_theft_count >= 2:
                theft_confidences = [c for p, c in theft_preds]
                avg_confidence = np.mean(theft_confidences)
                return 1, min(avg_confidence, 0.95)  # 限制最高置信度
            
            # 情况2: 最近一次是高置信度盗窃，且之前也有盗窃预测
            elif recent_preds[-1][0] == 1 and recent_preds[-1][1] > 0.8 and theft_count >= 1:
                return 1, recent_preds[-1][1] * 0.9
            
            # 情况3: 连续的盗窃预测（即使置信度不是很高）
            elif theft_count >= 2 and all(c > 0.6 for p, c in theft_preds):
                theft_confidences = [c for p, c in theft_preds]
                avg_confidence = np.mean(theft_confidences)
                return 1, avg_confidence * 0.85
        
        # 默认情况：判定为正常
        if normal_preds:
            normal_confidences = [c for p, c in normal_preds]
            avg_normal_conf = np.mean(normal_confidences)
            return 0, max(0.6, avg_normal_conf)
        else:
            # 如果没有正常预测，但也不满足盗窃条件，返回低置信度的正常
            return 0, 0.6
    
    def test_video_file(self, video_path: str, output_path: str = None):
        """测试本地视频文件"""
        print(f"Testing video file: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {video_path}")
            return
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # 计算扩展画布尺寸（右侧添加信息面板）
        panel_width = 400
        canvas_width = width + panel_width
        canvas_height = height
        
        # 设置输出视频（如果指定）
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))
        
        frame_count = 0
        theft_detections = []
        
        # 存储最新的预测结果，用于在所有帧上显示
        latest_prediction = {
            'pred_class': 0,
            'confidence': 0.0,
            'probs': [1.0, 0.0],
            'smooth_class': 0,
            'smooth_conf': 0.0
        }
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                timestamp = frame_count / fps
                
                # 添加帧到处理器
                self.video_processor.add_frame(frame)
                
                # 每隔一定帧数进行一次预测
                if frame_count % max(1, fps // 3) == 0:  # 每0.33秒预测一次（更频繁）
                    clip = self.video_processor.get_clip()
                    if clip[0] is not None:
                        pred_class, confidence, probs = self.predict(clip)
                        smooth_class, smooth_conf = self.smooth_prediction(pred_class, confidence)
                        
                        # 更新最新预测结果
                        latest_prediction = {
                            'pred_class': pred_class,
                            'confidence': confidence,
                            'probs': probs,
                            'smooth_class': smooth_class,
                            'smooth_conf': smooth_conf
                        }
                        
                        if smooth_class == 1:  # 检测到盗窃
                            theft_detections.append({
                                'timestamp': timestamp,
                                'confidence': smooth_conf,
                                'frame': frame_count
                            })
                            print(f"THEFT DETECTED at {timestamp:.2f}s (frame {frame_count}), "
                                  f"confidence: {smooth_conf:.3f}")
                
                # 创建扩展画布（原视频 + 右侧信息面板）
                canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                
                # 将原始视频帧放在左侧
                canvas[:height, :width] = frame
                
                # 在右侧创建信息面板
                panel_x = width
                panel_color = (40, 40, 40)  # 深灰色背景
                canvas[:, panel_x:] = panel_color
                
                # 绘制面板边框
                cv2.line(canvas, (panel_x, 0), (panel_x, canvas_height), (255, 255, 255), 2)
                
                # 设置文本颜色
                color = (0, 0, 255) if latest_prediction['smooth_class'] == 1 else (0, 255, 0)
                text_color = (255, 255, 255)
                
                # 面板标题
                title_text = "THEFT DETECTION"
                cv2.putText(canvas, title_text, (panel_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, text_color, 2)
                
                # 主要预测结果
                result_text = f"Result: {self.class_names[latest_prediction['smooth_class']]}"
                cv2.putText(canvas, result_text, (panel_x + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)
                
                # 置信度
                smooth_conf_safe = latest_prediction['smooth_conf']
                if np.isnan(smooth_conf_safe) or np.isinf(smooth_conf_safe):
                    smooth_conf_safe = 0.0
                smooth_conf_safe = max(0.0, min(1.0, smooth_conf_safe))
                
                conf_text = f"Confidence: {smooth_conf_safe:.3f}"
                cv2.putText(canvas, conf_text, (panel_x + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, text_color, 1)
                
                # 置信度条
                bar_x = panel_x + 10
                bar_y = 110
                bar_width = int(300 * smooth_conf_safe)
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 300, bar_y + 20), (60, 60, 60), -1)  # 背景
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), color, -1)  # 填充
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 300, bar_y + 20), text_color, 1)  # 边框
                
                # 置信度百分比
                conf_percent = f"{smooth_conf_safe*100:.1f}%"
                cv2.putText(canvas, conf_percent, (bar_x + 310, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
                
                # 原始预测结果
                raw_text = f"Raw Prediction:"
                cv2.putText(canvas, raw_text, (panel_x + 10, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, text_color, 1)
                
                raw_result = f"{self.class_names[latest_prediction['pred_class']]} ({latest_prediction['confidence']:.3f})"
                cv2.putText(canvas, raw_result, (panel_x + 10, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
                
                # 类别概率
                probs = latest_prediction['probs']
                normal_prob = probs[0] if len(probs) > 0 else 0.0
                theft_prob = probs[1] if len(probs) > 1 else 0.0
                
                prob_title = "Class Probabilities:"
                cv2.putText(canvas, prob_title, (panel_x + 10, 220), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, text_color, 1)
                
                normal_text = f"Normal: {normal_prob:.3f}"
                cv2.putText(canvas, normal_text, (panel_x + 10, 245), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (0, 255, 0), 1)
                
                theft_text = f"Theft:  {theft_prob:.3f}"
                cv2.putText(canvas, theft_text, (panel_x + 10, 265), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (0, 0, 255), 1)
                
                # 时间和帧信息
                time_title = "Video Info:"
                cv2.putText(canvas, time_title, (panel_x + 10, 305), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, text_color, 1)
                
                time_text = f"Time: {timestamp:.1f}s"
                cv2.putText(canvas, time_text, (panel_x + 10, 325), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
                
                frame_text = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(canvas, frame_text, (panel_x + 10, 345), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
                
                # 进度条
                progress_ratio = frame_count / total_frames
                progress_bar_width = int(300 * progress_ratio)
                progress_y = 355
                cv2.rectangle(canvas, (bar_x, progress_y), (bar_x + 300, progress_y + 15), (60, 60, 60), -1)
                cv2.rectangle(canvas, (bar_x, progress_y), (bar_x + progress_bar_width, progress_y + 15), (255, 255, 0), -1)
                cv2.rectangle(canvas, (bar_x, progress_y), (bar_x + 300, progress_y + 15), text_color, 1)
                
                progress_text = f"{progress_ratio*100:.1f}%"
                cv2.putText(canvas, progress_text, (bar_x + 310, progress_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
                
                # 如果检测到盗窃，在面板上添加警告
                if latest_prediction['smooth_class'] == 1:
                    warning_y = canvas_height - 60
                    cv2.rectangle(canvas, (panel_x + 5, warning_y - 25), (canvas_width - 5, warning_y + 25), (0, 0, 255), -1)
                    cv2.putText(canvas, "⚠ THEFT ALERT ⚠", (panel_x + 15, warning_y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
                
                # 使用扩展画布替代原始帧
                frame = canvas
                
                # 写入输出视频
                if out:
                    out.write(frame)
                
                # 显示进度
                if frame_count % (fps * 5) == 0:  # 每5秒显示一次进度
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        finally:
            cap.release()
            if out:
                out.release()
        
        # 输出检测结果摘要
        print(f"\n=== Detection Summary ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Theft detections: {len(theft_detections)}")
        
        if theft_detections:
            print("\nDetection details:")
            for detection in theft_detections:
                print(f"  - Time: {detection['timestamp']:.2f}s, "
                      f"Frame: {detection['frame']}, "
                      f"Confidence: {detection['confidence']:.3f}")
    
    def test_rtsp_stream(self, rtsp_url: str, duration: int = 60):
        """测试RTSP流"""
        print(f"Testing RTSP stream: {rtsp_url}")
        print(f"Duration: {duration} seconds")
        
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"Error: Cannot connect to RTSP stream: {rtsp_url}")
            return
        
        # 获取流信息
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25  # 默认25fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Stream info: {width}x{height}, {fps} FPS")
        
        # 计算扩展画布尺寸（右侧添加信息面板）
        panel_width = 400
        canvas_width = width + panel_width
        canvas_height = height
        
        start_time = time.time()
        frame_count = 0
        theft_detections = []
        
        # 存储最新的预测结果，用于在所有帧上显示
        latest_prediction = {
            'pred_class': 0,
            'confidence': 0.0,
            'probs': [1.0, 0.0],
            'smooth_class': 0,
            'smooth_conf': 0.0
        }
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to read frame from stream")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                current_time = time.time() - start_time
                
                # 添加帧到处理器
                self.video_processor.add_frame(frame)
                
                # 每隔一定帧数进行一次预测
                if frame_count % max(1, fps // 3) == 0:  # 每0.33秒预测一次（更频繁）
                    clip = self.video_processor.get_clip()
                    if clip[0] is not None:
                        pred_class, confidence, probs = self.predict(clip)
                        smooth_class, smooth_conf = self.smooth_prediction(pred_class, confidence)
                        
                        # 更新最新预测结果
                        latest_prediction = {
                            'pred_class': pred_class,
                            'confidence': confidence,
                            'probs': probs,
                            'smooth_class': smooth_class,
                            'smooth_conf': smooth_conf
                        }
                        
                        if smooth_class == 1:  # 检测到盗窃
                            theft_detections.append({
                                'timestamp': current_time,
                                'confidence': smooth_conf,
                                'frame': frame_count
                            })
                            print(f"THEFT DETECTED at {current_time:.2f}s (frame {frame_count}), "
                                  f"confidence: {smooth_conf:.3f}")
                
                # 创建扩展画布（原视频 + 右侧信息面板）- 在每一帧上都绘制
                canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                
                # 将原始视频帧放在左侧
                canvas[:height, :width] = frame
                
                # 在右侧创建信息面板
                panel_x = width
                panel_color = (40, 40, 40)  # 深灰色背景
                canvas[:, panel_x:] = panel_color
                
                # 绘制面板边框
                cv2.line(canvas, (panel_x, 0), (panel_x, canvas_height), (255, 255, 255), 2)
                
                # 设置文本颜色
                color = (0, 0, 255) if latest_prediction['smooth_class'] == 1 else (0, 255, 0)
                text_color = (255, 255, 255)
                
                # 面板标题
                title_text = "THEFT DETECTION"
                cv2.putText(canvas, title_text, (panel_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, text_color, 2)
                
                # 主要预测结果
                result_text = f"Result: {self.class_names[latest_prediction['smooth_class']]}"
                cv2.putText(canvas, result_text, (panel_x + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)
                
                # 置信度
                smooth_conf_safe = latest_prediction['smooth_conf']
                if np.isnan(smooth_conf_safe) or np.isinf(smooth_conf_safe):
                    smooth_conf_safe = 0.0
                smooth_conf_safe = max(0.0, min(1.0, smooth_conf_safe))
                
                conf_text = f"Confidence: {smooth_conf_safe:.3f}"
                cv2.putText(canvas, conf_text, (panel_x + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, text_color, 1)
                
                # 置信度条
                bar_x = panel_x + 10
                bar_y = 110
                bar_width = int(300 * smooth_conf_safe)
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 300, bar_y + 20), (60, 60, 60), -1)  # 背景
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), color, -1)  # 填充
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 300, bar_y + 20), text_color, 1)  # 边框
                
                # 置信度百分比
                conf_percent = f"{smooth_conf_safe*100:.1f}%"
                cv2.putText(canvas, conf_percent, (bar_x + 310, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
                
                # 原始预测结果
                raw_text = f"Raw Prediction:"
                cv2.putText(canvas, raw_text, (panel_x + 10, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, text_color, 1)
                
                raw_result = f"{self.class_names[latest_prediction['pred_class']]} ({latest_prediction['confidence']:.3f})"
                cv2.putText(canvas, raw_result, (panel_x + 10, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
                
                # 类别概率
                probs = latest_prediction['probs']
                normal_prob = probs[0] if len(probs) > 0 else 0.0
                theft_prob = probs[1] if len(probs) > 1 else 0.0
                
                prob_title = "Class Probabilities:"
                cv2.putText(canvas, prob_title, (panel_x + 10, 220), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, text_color, 1)
                
                normal_text = f"Normal: {normal_prob:.3f}"
                cv2.putText(canvas, normal_text, (panel_x + 10, 245), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (0, 255, 0), 1)
                
                theft_text = f"Theft:  {theft_prob:.3f}"
                cv2.putText(canvas, theft_text, (panel_x + 10, 265), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (0, 0, 255), 1)
                
                # 时间和帧信息
                time_title = "Stream Info:"
                cv2.putText(canvas, time_title, (panel_x + 10, 305), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, text_color, 1)
                
                time_text = f"Time: {current_time:.1f}s"
                cv2.putText(canvas, time_text, (panel_x + 10, 325), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
                
                frame_text = f"Frame: {frame_count}"
                cv2.putText(canvas, frame_text, (panel_x + 10, 345), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
                
                fps_text = f"FPS: {fps}"
                cv2.putText(canvas, fps_text, (panel_x + 10, 365), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
                
                # 如果检测到盗窃，在面板上添加警告
                if latest_prediction['smooth_class'] == 1:
                    warning_y = canvas_height - 60
                    cv2.rectangle(canvas, (panel_x + 5, warning_y - 25), (canvas_width - 5, warning_y + 25), (0, 0, 255), -1)
                    cv2.putText(canvas, "⚠ THEFT ALERT ⚠", (panel_x + 15, warning_y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
                
                # 显示帧
                cv2.imshow('RTSP Stream - Theft Detection', canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 显示进度
                if frame_count % (fps * 10) == 0:  # 每10秒显示一次进度
                    elapsed = time.time() - start_time
                    print(f"Elapsed: {elapsed:.1f}s ({frame_count} frames processed)")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # 输出检测结果摘要
        elapsed_time = time.time() - start_time
        print(f"\n=== Detection Summary ===")
        print(f"Total time: {elapsed_time:.1f}s")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {frame_count / elapsed_time:.1f}")
        print(f"Theft detections: {len(theft_detections)}")
        
        if theft_detections:
            print("\nDetection details:")
            for detection in theft_detections:
                print(f"  - Time: {detection['timestamp']:.2f}s, "
                      f"Frame: {detection['frame']}, "
                      f"Confidence: {detection['confidence']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Test theft detection model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--rtsp', type=str, help='RTSP stream URL')
    parser.add_argument('--output', type=str, help='Output video path (for video file testing)')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration for RTSP stream testing (seconds)')
    
    args = parser.parse_args()
    
    if not args.video and not args.rtsp:
        print("Error: Please specify either --video or --rtsp")
        return
    
    # 创建检测器
    detector = TheftDetector(args.config, args.checkpoint)
    
    if args.video:
        # 测试本地视频文件
        detector.test_video_file(args.video, args.output)
    elif args.rtsp:
        # 测试RTSP流
        detector.test_rtsp_stream(args.rtsp, args.duration)


if __name__ == '__main__':
    main()