# SMIC 四元结构最小性检验：双真实数据集消融实验

本目录执行 `J=(S,M,I,C)` 四元结构的留一消融、结构替代和跨域预测基线比较。

## 数据集

### 数据集 1：纽约市绿色出租车

- 官方来源：NYC Taxi & Limousine Commission
- 数据：2015 年 1 月绿色出租车行程记录
- URL：<https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2015-01.parquet>
- 原始记录：1,508,493 条
- 主体映射：将 `PULocationID` 视为主体，选择出现频率最高的 40 个区域
- 初始资源：每个主体的行程计数，归一化为均值 1

### 数据集 2：UCI ElectricityLoadDiagrams20112014

- 官方来源：UCI Machine Learning Repository
- DOI：10.24432/C58C86
- URL：<https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip>
- 原始数据：370 个客户，15 分钟间隔，2011–2014 年，单位为 kW
- 主体映射：选择平均负荷最高的 40 个客户
- 初始资源：40 个客户的平均负荷，归一化为均值 1

原始数据文件没有上传 GitHub；代码会通过 README 中的官方 URL 下载。这样可以避免仓库包含约 250 MB 的原始压缩文件。

## 实验协议

- 主体数：40
- 时间步：200
- 随机种子：0–199，共 200 个
- 高斯扰动：均值 0，标准差 0.015
- 资源下限：0.02
- 每步统一增量：0.004
- 所有条件在同一 seed 内共享同一初始资源、噪声和统一增量
- 交互：随机打乱、两两配对，资源较多者向较少者转移，转移量为 `min(0.8×|差值|, cap)`
- Full SMIC：`cap=0.01`

### 条件

- `Full_SMIC`：S、M、I、C 全部保留
- `remove_M`：主体模块同质化
- `remove_I`：无主体间交互，仅独立噪声和增量演化
- `remove_C`：保留交互但取消转移上限
- `remove_S`：不保留历史状态，重置为当前均匀可行状态
- `SMI`：删除 C，使用无上限交互
- `SIC`：删除 M，主体同质化并保留约束
- `MIC`：删除 S，使用当前均匀状态并保留交互和约束
- `MI`：删除 S 和 C
- `M_only`：只保留同质化模块状态

这些是明确的操作化模型，不等同于证明所有现实系统都只能用这四个元素解释。

## 主要消融结果

### 出租车数据：最终 Gini

| 条件 | 均值 | 标准差 | 95% CI |
|---|---:|---:|---:|
| Full_SMIC | 0.008232 | 0.001255 | [0.008057, 0.008407] |
| remove_M | 0.004546 | 0.000532 | [0.004472, 0.004620] |
| remove_I | 0.203998 | 0.011461 | [0.202400, 0.205596] |
| remove_C | 0.008035 | 0.001030 | [0.007892, 0.008179] |
| remove_S | 0.004546 | 0.000532 | [0.004472, 0.004620] |
| SMI | 0.008035 | 0.001030 | [0.007892, 0.008179] |
| SIC | 0.004546 | 0.000532 | [0.004472, 0.004620] |
| MIC | 0.004546 | 0.000532 | [0.004472, 0.004620] |
| MI | 0.004546 | 0.000532 | [0.004472, 0.004620] |
| M_only | 0.004546 | 0.000532 | [0.004472, 0.004620] |

Friedman：`χ²=1759.495814`，`p<1×10^-34`（SciPy 数值输出为 0）。

### 电力数据：最终 Gini

| 条件 | 均值 | 标准差 | 95% CI |
|---|---:|---:|---:|
| Full_SMIC | 0.164305 | 0.006101 | [0.163454, 0.165156] |
| remove_M | 0.004546 | 0.000532 | [0.004472, 0.004620] |
| remove_I | 0.350473 | 0.012525 | [0.348726, 0.352219] |
| remove_C | 0.008035 | 0.001030 | [0.007892, 0.008179] |
| remove_S | 0.004546 | 0.000532 | [0.004472, 0.004620] |
| SMI | 0.008035 | 0.001030 | [0.007892, 0.008179] |
| SIC | 0.004546 | 0.000532 | [0.004472, 0.004620] |
| MIC | 0.004546 | 0.000532 | [0.004472, 0.004620] |
| MI | 0.004546 | 0.000532 | [0.004472, 0.004620] |
| M_only | 0.004546 | 0.000532 | [0.004472, 0.004620] |

Friedman：`χ²=1800.000000`，`p<1×10^-34`（SciPy 数值输出为 0）。

## Full SMIC 与各条件的配对检验

完整结果见 `statistical_tests.csv`。显著性并不自动等于理论最小性；还必须考虑操作化是否公平以及消融模型是否被过度简化。

最稳定的结果是：删除 I 后最终 Gini 显著升高，且在两个数据集上都出现很大的负向 `Cohen's d_z`。这支持 **I 在当前资源转移任务中具有条件性必要性**。

删除 C 的结果在两个数据集方向一致，但出租车数据中的效应较小（`p=0.032936`，`d_z=0.148671`），电力数据中的效应较大。这个差异说明 C 的影响依赖数据域和初始状态。

删除 M 或 S 后结果显著变化，但当前实现将相应简化模型操作化为主体同质化或均匀状态重置，因此不能据此宣称 M 或 S 已被普遍证明不可替代。需要后续等容量替代模型和信息匹配实验。

## 跨域预测比较

| 数据集 | 模型 | RMSE | MAE | R² | 状态转移准确率 |
|---|---|---:|---:|---:|---:|
| taxi | SMIC | 744.280 | 566.994 | 0.665 | 0.785 |
| taxi | system_dynamics | 1335.355 | 1091.391 | -0.079 | 0.537 |
| taxi | ARIMA | 1621.392 | 1357.640 | -0.591 | 0.537 |
| taxi | random_walk | 361.590 | 274.738 | 0.921 | 0.785 |
| electricity | SMIC | 631.768 | 560.323 | 0.084 | 0.749 |
| electricity | system_dynamics | 705.311 | 644.378 | -0.142 | 0.259 |
| electricity | ARIMA | 941.606 | 710.320 | -1.036 | 0.485 |
| electricity | random_walk | 221.662 | 157.424 | 0.887 | 0.749 |

这是本次实验最重要的负面结果之一：**SMIC 没有在 RMSE、MAE 或 R² 上击败随机游走基线。** 因此，本实验不能支持“SMIC 普遍优于传统预测基线”的结论。它只支持：在本操作化下，I 的交互结构对资源状态轨迹有明显影响；四元结构的普遍最小性和跨域预测增益仍未证明。

## 证伪性结论

当前结果支持以下较窄结论：

> 在两个异质真实数据集、固定主体选择和指定资源转移动力学下，完整 SMIC 与各消融模型存在显著差异；其中 I 的删除在两个数据集上产生最稳定、最大的轨迹和 Gini 变化，C 的影响具有条件性，M 和 S 的结果依赖当前简化操作化。

当前结果不支持以下强结论：

- 不支持 S、M、I、C 都已经被证明是普遍不可替代的；
- 不支持 SMIC 在所有预测任务上优于 ARIMA、系统动力学或随机游走；
- 不支持真实出租车或电力系统中的因果机制已经被识别；
- 不支持四元结构已经完成数学意义上的最小性证明。

下一步应做：等容量替代模型、信息匹配的置换实验、N=2/N=3 解析枚举、跨机制 I1–I4 复制，以及避免把“删除模型变成均匀模型”所造成的容量差异误判为元素必要性。

## 复现

```bash
pip install numpy pandas pyarrow scipy matplotlib
curl -L -o data/green_tripdata_2015-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2015-01.parquet
curl -L -o data/electricityloaddiagrams20112014.zip \
  https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip -o data/electricityloaddiagrams20112014.zip -d data
python run_ablation.py --root .
```

## 文件

- `run_ablation.py`：完整 Python 3.11 兼容代码
- `ablation_results.csv`：两个数据集的最终 Gini、CI 和轨迹误差
- `statistical_tests.csv`：Friedman、Wilcoxon、Cohen's `d_z`
- `cross_domain_metrics.csv`：SMIC、系统动力学、ARIMA、随机游走比较
- `figure1_ablation_trajectories.png`、`figure2_ablation_boxplot.png`：规定名称的出租车图
- `figure1_ablation_trajectories_taxi.png`、`figure2_ablation_boxplot_taxi.png`：出租车图
- `figure1_ablation_trajectories_electricity.png`、`figure2_ablation_boxplot_electricity.png`：电力图
- `figure3_cross_domain.png`：跨域 RMSE 比较
- `experiment_manifest.json`：数据和实验参数记录
