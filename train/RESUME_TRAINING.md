# 断点续训使用指南

## 概述

现在您的训练系统已经支持完整的断点续训功能！即使电脑断电或训练意外中断，您也可以从上次的进度继续训练。

## 功能特点

✅ **自动保存**: 每个epoch结束后自动保存checkpoint  
✅ **自动恢复**: 重新运行训练时自动检测并加载最新checkpoint  
✅ **完整状态**: 保存模型权重、优化器状态、epoch进度、最佳准确率等  
✅ **最佳模型**: 单独保存性能最好的模型  
✅ **管理工具**: 提供checkpoint管理和清理工具  

## 文件说明

### Checkpoint文件类型

1. **`latest_checkpoint.pth`** - 最新的训练checkpoint，用于断点续训
2. **`best_model.pth`** - 验证准确率最高的模型，用于推理部署

### 保存内容

每个checkpoint包含：
- `epoch`: 当前epoch编号
- `model_state_dict`: 模型参数
- `optimizer_state_dict`: 优化器状态
- `best_val_accuracy`: 历史最佳验证准确率
- `train_loss`: 当前epoch训练损失
- `val_loss`: 当前epoch验证损失
- `val_accuracy`: 当前epoch验证准确率

## 使用方法

### 1. 正常训练
```bash
python train.py --config configs/config.yaml
```

### 2. 断点续训
如果训练中断，直接重新运行相同命令即可：
```bash
python train.py --config configs/config.yaml
```
程序会自动检测并加载 `checkpoints/latest_checkpoint.pth`

### 3. 查看checkpoint信息
```bash
# 列出所有checkpoint
python resume_training.py --list

# 查看特定checkpoint详情
python resume_training.py --info checkpoints/latest_checkpoint.pth
```

### 4. 备份重要checkpoint
```bash
python resume_training.py --backup checkpoints/latest_checkpoint.pth
```

### 5. 清理旧checkpoint
```bash
# 只保留最新5个checkpoint
python resume_training.py --clean

# 自定义保留数量
python resume_training.py --clean --keep 10
```

## 训练输出示例

### 首次训练
```
No checkpoint found, starting training from scratch

--- Starting Training ---
Epoch 1 | Average Training Loss: 0.6234
Epoch 1 | Validation Loss: 0.5123 | Validation Accuracy: 0.7500
Checkpoint saved to checkpoints/latest_checkpoint.pth
-> New best model saved to checkpoints/best_model.pth with accuracy: 0.7500
```

### 断点续训
```
Loading checkpoint from checkpoints/latest_checkpoint.pth
Resuming training from epoch 15
Previous best validation accuracy: 0.8750

--- Starting Training ---
Epoch 15 | Average Training Loss: 0.2341
Epoch 15 | Validation Loss: 0.2456 | Validation Accuracy: 0.8900
Checkpoint saved to checkpoints/latest_checkpoint.pth
-> New best model saved to checkpoints/best_model.pth with accuracy: 0.8900
```

## 注意事项

### ⚠️ 重要提醒

1. **不要删除 `latest_checkpoint.pth`** - 这是续训的关键文件
2. **定期备份** - 重要的checkpoint建议手动备份
3. **磁盘空间** - checkpoint文件较大，注意磁盘空间
4. **配置一致** - 续训时使用相同的配置文件

### 🔧 故障排除

**问题**: 找不到checkpoint文件  
**解决**: 检查 `checkpoints/` 目录是否存在 `latest_checkpoint.pth`

**问题**: 加载checkpoint失败  
**解决**: 
- 检查PyTorch版本兼容性
- 确认checkpoint文件未损坏
- 尝试从备份文件恢复

**问题**: 想从特定epoch重新开始  
**解决**: 
- 备份当前checkpoint
- 删除或重命名 `latest_checkpoint.pth`
- 从 `best_model.pth` 开始新的训练

## 高级用法

### 手动指定checkpoint
如果需要从特定checkpoint恢复，可以修改代码中的checkpoint路径，或者将目标checkpoint重命名为 `latest_checkpoint.pth`。

### 迁移checkpoint
如果需要在不同机器间迁移训练：
1. 复制整个 `checkpoints/` 目录
2. 确保目标机器有相同的Python环境
3. 运行训练命令即可继续

### 监控训练进度
使用TensorBoard查看训练曲线：
```bash
tensorboard --logdir=runs
```

## 最佳实践

1. **定期备份**: 每隔几个epoch手动备份重要checkpoint
2. **监控磁盘**: checkpoint文件会占用较多空间，定期清理
3. **记录配置**: 保存每次训练使用的配置文件
4. **版本管理**: 重要的模型版本建议用git管理

---

现在您可以放心训练了！即使遇到断电等意外情况，也能无缝恢复训练进度。 