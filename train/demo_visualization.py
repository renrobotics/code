#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化演示脚本
展示测试脚本中的可视化功能
"""

import cv2
import numpy as np


def create_demo_frame(width=640, height=480):
    """创建一个演示帧"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加一些背景图案
    cv2.rectangle(frame, (50, 50), (width-50, height-50), (30, 30, 30), -1)
    cv2.circle(frame, (width//2, height//2), 100, (60, 60, 60), -1)
    
    return frame


def draw_enhanced_overlay(frame, class_name, confidence, raw_confidence, probs, 
                         timestamp, frame_count, is_theft=False):
    """绘制增强的可视化覆盖层"""
    height, width = frame.shape[:2]
    
    # 颜色设置
    color = (0, 0, 255) if is_theft else (0, 255, 0)  # 红色/绿色
    
    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # 主要预测结果
    text = f"{class_name}: {confidence:.3f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, color, 2)
    
    # 显示原始置信度（未平滑）
    raw_class = "Theft" if raw_confidence > 0.5 else "Normal"
    raw_text = f"Raw: {raw_class} ({raw_confidence:.3f})"
    cv2.putText(frame, raw_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    # 显示两个类别的概率
    normal_prob = probs[0] if len(probs) > 0 else 0.0
    theft_prob = probs[1] if len(probs) > 1 else 0.0
    prob_text = f"Normal: {normal_prob:.3f} | Theft: {theft_prob:.3f}"
    cv2.putText(frame, prob_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    # 绘制置信度条
    bar_width = int(300 * confidence)
    cv2.rectangle(frame, (10, 85), (310, 105), (50, 50, 50), -1)  # 背景
    cv2.rectangle(frame, (10, 85), (10 + bar_width, 105), color, -1)  # 填充
    cv2.rectangle(frame, (10, 85), (310, 105), (255, 255, 255), 2)  # 边框
    
    # 置信度百分比
    conf_percent = f"{confidence*100:.1f}%"
    cv2.putText(frame, conf_percent, (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    # 添加时间戳
    time_text = f"Time: {timestamp:.1f}s | Frame: {frame_count}"
    cv2.putText(frame, time_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    return frame


def main():
    """演示可视化功能"""
    print("可视化演示 - 按任意键切换场景，按ESC退出")
    
    # 创建演示场景
    scenarios = [
        {
            "class_name": "Normal",
            "confidence": 0.892,
            "raw_confidence": 0.856,
            "probs": [0.892, 0.108],
            "is_theft": False,
            "description": "正常行为检测"
        },
        {
            "class_name": "Theft", 
            "confidence": 0.734,
            "raw_confidence": 0.689,
            "probs": [0.266, 0.734],
            "is_theft": True,
            "description": "盗窃行为检测"
        },
        {
            "class_name": "Normal",
            "confidence": 0.623,
            "raw_confidence": 0.578,
            "probs": [0.623, 0.377],
            "is_theft": False,
            "description": "低置信度正常行为"
        },
        {
            "class_name": "Theft",
            "confidence": 0.945,
            "raw_confidence": 0.912,
            "probs": [0.055, 0.945],
            "is_theft": True,
            "description": "高置信度盗窃检测"
        }
    ]
    
    scenario_idx = 0
    frame_count = 1
    timestamp = 0.0
    
    while True:
        # 创建演示帧
        frame = create_demo_frame()
        
        # 获取当前场景
        scenario = scenarios[scenario_idx]
        
        # 绘制可视化覆盖层
        frame = draw_enhanced_overlay(
            frame,
            scenario["class_name"],
            scenario["confidence"],
            scenario["raw_confidence"],
            scenario["probs"],
            timestamp,
            frame_count,
            scenario["is_theft"]
        )
        
        # 添加场景描述
        desc_text = f"场景 {scenario_idx + 1}/4: {scenario['description']}"
        cv2.putText(frame, desc_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 0), 2)
        
        # 添加操作提示
        help_text = "按任意键切换场景，按ESC退出"
        cv2.putText(frame, help_text, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # 显示帧
        cv2.imshow('Theft Detection - Visualization Demo', frame)
        
        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC键
            break
        else:
            # 切换到下一个场景
            scenario_idx = (scenario_idx + 1) % len(scenarios)
            frame_count += 30  # 模拟30帧后的检测
            timestamp += 1.0   # 模拟1秒后的检测
    
    cv2.destroyAllWindows()
    print("演示结束")


if __name__ == '__main__':
    main() 