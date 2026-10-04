# Real-data multi-agent resource allocation experiment

本目录实现附录 D 多主体资源分配实验的真实数据版，严格遵循用户指定的实验协议。

## 数据来源

使用纽约市 Taxi & Limousine Commission 官方绿色出租车行程记录：

- 数据：2015 年 1 月绿色出租车记录
- 官方下载地址：<https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2015-01.parquet>
- 记录数：1,508,493
- 数据说明：每行代表一条绿色出租车行程，包含 `PULocationID`、`DOLocationID`、`trip_distance` 等字段。
- 本实验按照任务要求，将 `PULocationID` 作为主体识别变量。

由于真实数据的 `PULocationID` 超过 40 个，实验固定选择该月份出现次数最多的 40 个 `PULocationID` 作为主体，并在 `data_selection.json` 中记录主体 ID、原始行程数和归一化后的初始资源。初始资源为各主体行程数除以 40 个主体行程数的均值，因此均值为 1。

原始 Parquet 文件不随 Git 提交，以避免重复占用仓库空间；运行脚本时会按照上述官方 URL 下载数据。

## 固定实验协议

- 主体数：`N=40`
- 时间步：`T=200`
- 随机种子：`seed=0,...,199`
- 扰动：每步高斯噪声 `N(0, 0.015)`
- 资源下限：`0.02`
- 每步统一增量：`0.004`
- 所有条件在同一 seed 内共享相同初始资源、扰动序列和统一增量

### 三个机学约束条件

- `fixed`：单步转移上限 `0.01`
- `unconstrained`：单步转移上限 `0.20`
- `adaptive`：上限 `min(0.08, 0.01 + 0.25 × Gini)`

每一步随机打乱主体、两两配对；每对按“较富者向较贫者”转移，转移量为 `min(0.8 × |差值|, cap)`。

### 三个基线条件

- `random`：随机配对、随机方向和随机转移量
- `uniform`：`r += 0.01 × (mean-r)`
- `greedy`：每步从最富主体向最贫主体转移

## 统计检验

- 六条件总体比较：Friedman 检验
- 预先指定的六个配对比较：配对 Wilcoxon 符号秩检验
- 效应量：配对 Cohen's `d_z`
- 置信区间：最终 Gini 均值的 95% t 置信区间

## 运行方法

```bash
pip install numpy pandas pyarrow scipy matplotlib
python run_experiment.py \
  --data green_tripdata_2015-01.parquet \
  --out .
```

也可以直接使用官方 URL 下载后运行：

```bash
curl -L -o green_tripdata_2015-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2015-01.parquet
python run_experiment.py --data green_tripdata_2015-01.parquet --out .
```

## 实际结果

| 条件 | 最终 Gini 均值 | 标准差 | 95% CI |
|---|---:|---:|---:|
| fixed | 0.008341 | 0.001256 | [0.008166, 0.008516] |
| unconstrained | 0.007991 | 0.001046 | [0.007845, 0.008137] |
| adaptive | 0.007689 | 0.001118 | [0.007533, 0.007845] |
| random | 0.353304 | 0.032498 | [0.348773, 0.357836] |
| uniform | 0.041567 | 0.004992 | [0.040871, 0.042263] |
| greedy | 0.013347 | 0.001897 | [0.013083, 0.013612] |

Friedman 检验：`χ²=922.162857`，`p=4.249347×10^-197`。

### 预先指定配对检验

| 比较 | Wilcoxon statistic | p 值 | Cohen's d_z |
|---|---:|---:|---:|
| fixed vs unconstrained | 7382.0 | 0.001132 | 0.251849 |
| fixed vs adaptive | 57.0 | 3.38×10^-34 | 1.598508 |
| adaptive vs unconstrained | 7009.0 | 0.000207 | -0.250738 |
| fixed vs random | 0.0 | 1.44×10^-34 | -10.663144 |
| fixed vs uniform | 0.0 | 1.44×10^-34 | -6.782423 |
| fixed vs greedy | 6.0 | 1.57×10^-34 | -2.442799 |

## 解释与限制

本结果必须按指定实验协议如实报告。它**不复现**此前合成实验中 fixed Gini 约 0.45–0.50 的预期范围。主要原因是当前协议中的统一增量、资源下限和 200 步运行共同造成了明显的均衡化动力学；这不是代码为了贴合预期而修改参数后的结果。

本实验使用真实出租车记录来定义初始主体资源，但后续 200 步是按指定的资源转移仿真规则运行。因此，它是“真实数据初始化 + 规则化动态实验”，不是对真实出租车调度行为的因果识别。`PULocationID` 是区域/位置 ID，将其称为主体是本研究的建模映射，而非出租车司机个体识别。

此外，`trip_distance` 和行程时间不进入当前仿真更新规则，因为任务设计明确规定初始资源由行程数量构造。若论文要主张真实交通资源分配机制，需要另做含 OD 流、时间、距离和供需变量的观测性验证。

## 文件

- `run_experiment.py`：完整 Python 3.11 兼容实验代码
- `final_summary.csv`：6 条件最终 Gini 均值、标准差和 95% CI
- `statistical_tests.csv`：Friedman、Wilcoxon 和 Cohen's `d_z`
- `figure1_trajectories_and_boxplot.png`：轨迹与终态箱线图
- `figure2_bar_comparison.png`：终态均值和 95% CI 柱状图
- `data_selection.json`：真实数据主体选择与初始资源记录
- `experiment_manifest.json`：实验参数和数据来源清单
