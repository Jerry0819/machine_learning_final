# 家庭用电多变量时间序列预测实验报告

> GitHub 链接：[TODO：提交前请替换为自己的仓库链接。](https://github.com/Jerry0819/machine_learning_final)

## 1. 问题介绍

随着智能家居和物联网技术的发展，家庭电力消耗预测可以帮助用户理解自身用电行为，也可以辅助电网进行负荷调度和节能管理。本实验使用 UCI Machine Learning Repository 中的 Individual household electric power consumption 数据集，对法国 Sceaux 一户家庭的用电数据进行建模，并预测未来每日总有功功率 `global_active_power`。

原始电力数据以分钟为粒度，时间范围为 2006-12 到 2010-11，包含总有功功率、无功功率、电压、电流以及三个分表能耗等变量。实验任务为：基于过去 90 天的多变量日级序列，分别预测未来 90 天和未来 365 天的 `global_active_power` 曲线。短期预测和长期预测分别训练模型，不共享输出层参数。

本实验比较三类模型：

1. LSTM 模型。
2. Transformer 模型。
3. 自提出的改进模型 MSConvTransformer。

评价指标为 MSE 和 MAE。每种模型、每个预测 horizon 均进行 5 次实验，报告均值和标准差。

## 2. 数据处理

### 2.1 电力数据日级聚合

原始分钟级电力数据来自：

- `individual+household+electric+power+consumption/household_power_consumption.txt`

预处理代码位于：

- `solution/preprocess.py`

电力变量按作业要求进行日级聚合：

| 变量 | 聚合方式 |
|---|---|
| `global_active_power` | 按天求和 |
| `global_reactive_power` | 按天求和 |
| `sub_metering_1` | 按天求和 |
| `sub_metering_2` | 按天求和 |
| `sub_metering_3` | 按天求和 |
| `voltage` | 按天求均值 |
| `global_intensity` | 按天求均值 |

同时计算未被三个分表覆盖的剩余能耗：

```text
sub_metering_remainder =
global_active_power * 1000 / 60
- sub_metering_1
- sub_metering_2
- sub_metering_3
```

为避免首尾不完整日期影响日总量统计，预处理时保留每日有效分钟数不少于 1000 的日期。缺失日期采用时间插值、前向填充和后向填充处理。最终得到 1441 条日级样本，时间范围为 2006-12-17 到 2010-11-26。

### 2.2 气象站选择与天气特征

电力数据采集地为法国 Sceaux。为了选择更贴近采集地点的天气数据，本实验使用 `solution/locate_station_id.py` 在周边气象文件中寻找距离 Sceaux 最近且数据覆盖较完整的气象站。

脚本中使用 Sceaux 近似坐标：

```text
latitude = 48.7788
longitude = 2.2906
```

并对候选气象站使用 Haversine 距离排序。最近且覆盖充分的站点为：

| NUM_POSTE | NOM_USUEL | 距离 Sceaux | 有效月份 |
|---:|---|---:|---:|
| 94034001 | FRESNES | 3.46 km | 48 |

因此，`preprocess.py` 默认使用 `station_id=94034001` 合并天气特征。天气变量来自月度基础气候数据，按月份合并到每日样本中：

| 变量 | 含义 |
|---|---|
| `RR` | 月累计降水量 |
| `NBJRR1` | 当月日降水量大于等于 1 mm 的天数 |
| `NBJRR5` | 当月日降水量大于等于 5 mm 的天数 |
| `NBJRR10` | 当月日降水量大于等于 10 mm 的天数 |
| `NBJBROU` | 当月雾日数 |

### 2.3 时间特征与监督学习样本

除电力和天气变量外，预处理还加入周期时间特征：

- `dow_sin`, `dow_cos`
- `month_sin`, `month_cos`
- `doy_sin`, `doy_cos`

训练样本构造方式为滑动窗口：

```text
input length = 90 days
output horizon = 90 days or 365 days
```

训练/测试按时间顺序切分，比例为 70% / 30%。模型训练时再从训练窗口尾部切分 15% 作为验证集，用于 early stopping 和学习率调度。特征标准化只在训练时间段上拟合，避免测试集信息泄漏。

## 3. 模型方法

### 3.1 LSTM 模型

LSTM 代码位于：

- `solution/train_lstm.py`

LSTM 模型使用两层 LSTM 编码 90 天输入序列。为了增强对重要时间步的利用，模型在 LSTM 输出序列上加入轻量注意力池化，并将最后隐藏状态与注意力上下文拼接后输入 MLP，直接输出未来完整 horizon 的预测结果。

主要训练设置：

| 参数 | 值 |
|---|---:|
| hidden size | 96 |
| num layers | 2 |
| dropout | 0.35 |
| batch size | 64 |
| learning rate | 5e-4 |
| weight decay | 5e-4 |
| loss | SmoothL1Loss |
| early stopping | validation loss |

LSTM 的优点是递归结构对连续趋势具有较强平滑归纳偏置，在长 horizon 预测中表现较稳定。

### 3.2 Transformer 模型

Transformer 代码位于：

- `solution/train_transformer.py`

Transformer 模型首先对输入特征做线性投影，并加入正弦位置编码，然后通过 Transformer Encoder 建模 90 天输入窗口内的全局依赖。输出阶段拼接最后一个 token 表征和全局平均池化表征，再通过 MLP 直接预测未来 90 天或 365 天曲线。

经过验证集调参后，主要参数如下：

| 参数 | 值 |
|---|---:|
| d_model | 96 |
| nhead | 4 |
| num layers | 2 |
| dropout | 0.30 |
| batch size | 64 |
| learning rate | 4e-4 |
| weight decay | 1e-3 |
| loss | SmoothL1Loss |

Transformer 的优势是自注意力可以直接建立任意两个时间步之间的联系，在 90 天短期预测上效果较好。

### 3.3 改进模型：MSConvTransformer

改进模型代码位于：

- `solution/train_improved.py`

详细说明位于：

- `solution/third_question_model.md`

本实验提出的改进模型命名为 MSConvTransformer，即 Multi-Scale Causal Convolution Transformer。模型结构为：

1. 输入 LayerNorm。
2. 线性投影到隐空间。
3. 门控多尺度因果卷积模块。
4. Transformer Encoder。
5. last / mean / max pooling 拼接。
6. MLP 多步预测头。

多尺度因果卷积使用三条并行深度可分离卷积分支：

| kernel size | dilation | 作用 |
|---:|---:|---|
| 3 | 1 | 捕捉短期连续波动 |
| 5 | 2 | 捕捉中等尺度用电变化 |
| 7 | 3 | 扩大局部感受野 |

初版改进模型使用强前置卷积，将所有输入先由卷积模块重编码。实验发现该设计在长 horizon 上容易过度强调局部波动，削弱长期趋势建模。最终版本改为门控残差结构：

```text
output = x + sigmoid(gate) * LocalConv(x)
```

这样 Transformer 主干始终保留，卷积特征只作为可学习的局部模式补充。当局部模式对预测有帮助时，模型可以增大卷积分支权重；当局部扰动不稳定时，门控机制可以降低卷积影响。

该方法有效的原因如下：

- 家庭用电序列既有连续数日的局部波动，也有跨周、跨月的长期依赖。
- 卷积层通过局部参数共享捕捉短期模式，在小样本时间序列上比纯注意力更稳定。
- 因果卷积避免使用未来信息，符合预测任务设置。
- Transformer 在局部特征之上建模全局依赖，弥补卷积感受野有限的问题。
- 多池化预测头同时保留最近状态、平均趋势和峰值响应，降低只取最后 token 的信息损失。
- SmoothL1Loss、dropout、weight decay、输入噪声和验证集早停共同降低过拟合风险。

## 4. 实验设置

所有模型均使用相同数据切分、输入长度和评价方式：

| 项目 | 设置 |
|---|---|
| 输入窗口 | 过去 90 天 |
| 预测长度 | 未来 90 天、未来 365 天 |
| 训练/测试划分 | 按时间 70% / 30% |
| 验证集 | 训练窗口尾部 15% |
| 标准化 | 仅使用训练时间段拟合 |
| 实验次数 | 5 |
| 指标 | MSE、MAE |
| 设备 | RTX 4060 CUDA |

运行命令示例：

```bash
python solution/preprocess.py --output-dir processed
python solution/train_lstm.py --runs 5 --horizons 90 365 --output-dir outputs/experiments
python solution/train_transformer.py --runs 5 --horizons 90 365 --output-dir outputs/experiments
python solution/train_improved.py --runs 5 --horizons 90 365 --epochs 100 --patience 18 --lr-patience 5 --output-dir outputs/experiments
```

说明：`outputs/smoke_*` 目录仅为冒烟测试结果，用于检查代码是否能跑通，不作为正式报告结果。正式结果均位于 `outputs/experiments`。

## 5. 实验结果

### 5.1 总体结果

| 模型 | Horizon | MSE mean | MSE std | MAE mean | MAE std |
|---|---:|---:|---:|---:|---:|
| LSTM | 90 | 145127.6094 | 5163.9516 | 291.3313 | 5.2561 |
| Transformer | 90 | 137588.5281 | 1544.6516 | 283.5467 | 2.6873 |
| MSConvTransformer | 90 | 144153.0750 | 7021.6859 | 289.8904 | 7.7289 |
| LSTM | 365 | 136775.2375 | 2336.3576 | 284.3333 | 2.6249 |
| Transformer | 365 | 145429.8281 | 5433.0014 | 294.8123 | 6.1032 |
| MSConvTransformer | 365 | 139826.9281 | 2704.3336 | 289.5892 | 3.3861 |

### 5.2 90 天预测分析

90 天预测中，Transformer 取得最佳结果：

```text
MSE = 137588.5281
MAE = 283.5467
```

这说明对于较短 horizon，90 天输入窗口内的全局注意力关系已经足够有效。Transformer 可以直接捕捉输入序列中不同时间步之间的相关性，且不需要经过递归传播，因此短期多步预测更有优势。

MSConvTransformer 在 90 天任务中优于 LSTM，但仍弱于 Transformer。原因可能是卷积模块虽然能提取局部波动，但短期预测中纯 Transformer 已经能够较好拟合局部和全局关系，额外卷积模块带来的收益不足以抵消模型方差增加。

### 5.3 365 天预测分析

365 天预测中，LSTM 取得最佳结果：

```text
MSE = 136775.2375
MAE = 284.3333
```

MSConvTransformer 排名第二，明显优于纯 Transformer：

```text
MSConvTransformer MAE = 289.5892
Transformer MAE = 294.8123
```

这说明对于长 horizon，单纯依赖 Transformer 注意力并不一定最稳。样本数量有限时，Transformer 更容易对训练窗口中的局部形状产生过拟合。LSTM 的递归和平滑归纳偏置对长期趋势更友好，因此在 365 天预测中表现最好。

MSConvTransformer 相比纯 Transformer 的提升来自门控多尺度卷积对局部模式的补充，以及残差门控对过强卷积偏置的抑制。它没有完全超过 LSTM，说明当前数据规模下，复杂混合模型的收益仍受样本量限制。

### 5.4 训练曲线和预测图

每次实验均保存训练曲线和预测曲线。可用于报告截图的文件示例：

| 模型 | Horizon | 文件 |
|---|---:|---|
| LSTM | 90 | `outputs/experiments/lstm_h90/run_1/loss_curve.png` |
| LSTM | 365 | `outputs/experiments/lstm_h365/run_1/prediction.png` |
| Transformer | 90 | `outputs/experiments/transformer_h90/run_1/loss_curve.png` |
| Transformer | 365 | `outputs/experiments/transformer_h365/run_1/prediction.png` |
| MSConvTransformer | 90 | `outputs/experiments/improved_h90/run_1/loss_curve.png` |
| MSConvTransformer | 365 | `outputs/experiments/improved_h365/run_1/prediction.png` |

## 6. 讨论

### 6.1 模型复杂度与性能的关系

实验表明，模型结构更复杂并不必然带来更高测试性能。MSConvTransformer 的参数量高于 LSTM 和 Transformer，但数据集日级样本只有 1441 条。对于这种小样本时间序列，复杂模型更容易出现方差增大或对验证窗口过拟合的问题。

在本实验中：

- Transformer 在 90 天预测上最优，说明短期多步预测更依赖窗口内全局关系建模。
- LSTM 在 365 天预测上最优，说明长 horizon 上平滑、稳定的递归结构仍然有优势。
- MSConvTransformer 在 365 天上优于纯 Transformer，说明局部卷积特征对长期预测有帮助，但仍未超过 LSTM。

### 6.2 为什么加入门控残差卷积

初版改进模型没有取得预期效果，主要原因是多尺度卷积被放在强前置位置，会先重编码全部输入。当卷积提取的局部波动与长期趋势不一致时，模型容易过度关注短期形状，导致长 horizon 泛化变差。

最终版本使用门控残差结构，将卷积变成对 Transformer 主干的补充，而不是替代。重跑实验后，365 天预测性能从初版的 MAE 约 300.8 提升到 289.6，说明该修改有效。

### 6.3 不足与改进方向

本实验仍有以下不足：

1. 天气数据为月度粒度，合并到每日样本后信息较粗，难以解释每日用电波动。
2. 总样本数量有限，深度模型的容量不能过大。
3. 当前采用直接多步预测，长 horizon 输出维度较大，训练难度较高。
4. 验证集和测试集时间段分布可能不完全一致，调参结果存在一定不确定性。

后续可以尝试：

- 使用更细粒度的日级天气数据。
- 引入节假日、工作日等外部特征。
- 对 365 天预测采用分段输出或多尺度输出头。
- 使用模型集成，将 LSTM 的长期稳定性与 Transformer 的短期表达能力结合。

## 7. 结论

本实验完成了家庭用电多变量时间序列预测任务，分别实现并比较了 LSTM、Transformer 和自提出的 MSConvTransformer。实验结果显示，不同模型在不同预测长度上具有不同优势：

- 90 天短期预测：Transformer 表现最好。
- 365 天长期预测：LSTM 表现最好。
- 改进模型 MSConvTransformer 在 365 天预测上明显优于纯 Transformer，但仍略低于 LSTM。

这说明家庭用电预测不仅需要全局依赖建模，也需要局部模式提取和合适的正则化。对于样本量有限的日级时间序列，模型复杂度需要谨慎控制；合理的归纳偏置往往比单纯增加模型容量更重要。

## 参考文献

[1] Dheeru Dua and Casey Graff. UCI Machine Learning Repository: Individual household electric power consumption. https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

[2] Data.gouv.fr. Donnees climatologiques de base mensuelles. https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles

[3] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.

[4] Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS. https://arxiv.org/abs/1706.03762

[5] Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. https://arxiv.org/abs/1803.01271

[6] Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI. https://arxiv.org/abs/2012.07436

## 工具使用声明

报告撰写和代码整理过程中使用了 ChatGPT / Codex 辅助生成文本、整理代码和调试实验。实验数据处理、模型训练和指标统计均基于本项目代码在本地环境运行得到。
