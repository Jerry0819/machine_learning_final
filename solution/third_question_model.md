# 第三问：改进模型 MSConvTransformer

## 模型结构

本题的改进模型命名为 **MSConvTransformer**，用于基于过去 90 天多变量日级特征直接预测未来 90 天或 365 天的 `global_active_power` 曲线。模型由四部分组成：

1. 输入标准化层：对每个时间步的多变量特征做 `LayerNorm`。
2. 门控多尺度因果卷积模块：先将输入投影到隐空间，再使用 kernel size 为 3、5、7，dilation 为 1、2、3 的三条并行深度可分离因果卷积分支，提取不同时间尺度的局部变化模式。卷积输出通过可学习门控残差加入主干，避免局部卷积过强地破坏长期趋势。
3. Transformer 编码器：在卷积提取的局部表征上执行自注意力建模，捕捉 90 天输入窗口内较长距离的依赖关系。
4. 多池化预测头：拼接最后一个时间步、全局平均池化和全局最大池化的表征，再通过 MLP 一次性输出完整预测 horizon。

## 为什么有效

家庭用电序列同时具有短期局部波动和较长周期依赖。纯 LSTM 依赖递归记忆，长 horizon 预测时容易累积误差；纯 Transformer 能建模远距离关系，但在样本量较小时直接学习局部形状会更难，且容易受噪声影响。

MSConvTransformer 的设计动机如下：

- 多尺度因果卷积先提取局部用电模式，例如连续几天的负载升降、周内行为变化和异常峰值。卷积具有局部平滑和参数共享特性，在小样本时间序列上更稳。
- 因果卷积只使用当前及过去信息，避免时间泄漏，符合预测任务设置。
- Transformer 编码器在局部模式之上建模长距离依赖，适合处理 90 天输入窗口中跨周、跨月的相关性。
- mean/max/last pooling 同时保留最近状态、整体趋势和峰值响应，减少只取最后 token 带来的信息损失。
- 使用 SmoothL1Loss、权重衰减、dropout、输入噪声和验证集早停，降低异常点和小样本过拟合对泛化性能的影响。

初版多尺度卷积是强前置结构，会先用卷积重编码全部输入特征。实验发现这种设计在长 horizon 上容易引入过强的局部归纳偏置，导致模型更重视局部波动而削弱长期趋势建模。最终版本改为门控残差卷积：Transformer 主干始终保留，卷积特征只以可学习权重补充进去。因此模型可以在卷积有帮助时利用局部模式，在卷积不稳定时减少其影响，泛化表现更稳。

## 训练与评估

运行方式与前两问一致：

```bash
python train_improved.py --runs 5 --horizons 90 365
```

默认设置：

- input length: 90 days
- output horizons: 90 days and 365 days
- runs: 5
- metrics: MSE and MAE
- selection: validation loss early stopping
- outputs: `outputs/experiments/improved_h90` and `outputs/experiments/improved_h365`

每次实验会保存：

- `metrics.json`
- `metadata.json`
- `config.json`
- `history.csv`
- `loss_curve.png`
- `prediction.png`
- `prediction_sample_*.png`

每个 horizon 会保存 `summary.json`，其中包含 MSE/MAE 的均值和标准差。

## 实验结果

本次正式实验命令：

```bash
python train_improved.py --runs 5 --horizons 90 365 --epochs 100 --patience 18 --lr-patience 5 --output-dir ../outputs/experiments
```

结果如下：

| Horizon | MSE mean | MSE std | MAE mean | MAE std | Avg epochs |
|---:|---:|---:|---:|---:|---:|
| 90 | 144153.0750 | 7021.6859 | 289.8904 | 7.7289 | 31.0 |
| 365 | 139826.9281 | 2704.3336 | 289.5892 | 3.3861 | 37.4 |

结果文件：

- `outputs/experiments/improved_h90/summary.json`
- `outputs/experiments/improved_h365/summary.json`
- `outputs/experiments/improved_metrics.csv`

## 参考文献

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. https://arxiv.org/abs/1706.03762
2. Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv. https://arxiv.org/abs/1803.01271
3. Zhou, H., et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*. AAAI. https://arxiv.org/abs/2012.07436
