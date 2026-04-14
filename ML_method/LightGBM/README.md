# LightGBM 预测系统使用说明

本目录包含基于LightGBM的SpGEMM配置预测系统，用于自动选择最佳的tile尺寸和TC阈值组合。

## 目录结构

```
LightGBM/
├── quick_predict_model/          # 训练好的模型和参考结果
│   ├── model_tuned.txt           # LightGBM模型文件（8100棵树）
│   ├── predict_testmatrices.py   # 参考预测脚本
│   └── 100_testmatrices_results.csv  # 参考预测结果
├── predictResult_test100/        # 100个测试矩阵的预测和执行
│   ├── predict_test100.py        # Task 1: 预测脚本
│   ├── run_test100.sh            # Task 2: 执行脚本
│   ├── test100_result.csv        # 输出结果
│   └── log/                      # 运行日志
├── predictResult_test12/         # 12个测试矩阵的完整流程
│   ├── predict_test12.py         # Task 3: 完整流程脚本
│   ├── test12mtx.txt             # 测试矩阵列表
│   ├── test12_result.csv         # 输出结果
│   └── log/                      # 运行日志
└── README.md                     # 本文件
```

## 快速开始

### 前置要求

- Python 3.6+ (使用conda环境: FlexSpGEMM)
- 已安装: pandas, lightgbm, numpy
- NVIDIA A100 GPU（用于执行SpGEMM）
- CUDA环境

### 测试流程

#### 步骤1: 预测100个测试矩阵的最佳配置

```bash
cd predictResult_test100
./run_predict.sh
```

**说明**: `run_predict.sh` 会自动激活conda环境并运行预测脚本。

**输出**: `test100_result.csv`（200行：100个矩阵 × 2种模式AA/AAT）

**预期结果**:
- 执行时间: <1秒
- 平均决策时间: ~0.2ms/矩阵
- 生成200行预测结果（每个矩阵包含AA和AAT两种模式）

#### 步骤2: 执行SpGEMM计算并验证性能

```bash
cd predictResult_test100
chmod +x run_test100.sh
./run_test100.sh
```

**功能**: 
- 使用预测的配置在A100上执行实际的SpGEMM计算
- 提取运行时间、GFLOPS、csr2tile时间
- 更新`test100_result.csv`和`probe.csv`

**注意**: 
- 运行时间较长（约需数小时，取决于矩阵大小）
- 需要GPU可用
- 日志保存在`log/`目录

#### 步骤3: 测试12个矩阵的完整流程（含理论最佳比较）

```bash
cd predictResult_test12
./run_predict.sh
```

**说明**: `run_predict.sh` 会自动激活conda环境并运行完整流程。

**功能**:
- 预测最佳配置
- 执行SpGEMM计算
- 从prime_data提取理论最佳GFLOPS
- 计算预测性能占理论最佳的比率（gflops_ratio）

**测试矩阵**: hangGlider_4、webbase-1M、pkustk12、Goodwin_095、af_shell10、s3rmq4m1、rma10、nemeth12、TSOPF_FS_b300_c2、trans5、heart3、gupta3

## 输出文件说明

### test100_result.csv

| 列名 | 说明 |
|------|------|
| matrix_name | 矩阵名称 |
| gpu | GPU类型（A100） |
| mode | 乘法模式（AA或AAT） |
| pred_combo | 预测的配置（如"16x16_0/8"） |
| runtime_ms | 运行时间（毫秒） |
| gflops | 性能（GFLOPS） |
| csr2tile_ms | CSR转Tile时间（毫秒） |

### test12_result.csv

在test100_result.csv的基础上增加：

| 列名 | 说明 |
|------|------|
| best_combo | 理论最佳配置 |
| best_gflops | 理论最佳GFLOPS |
| gflops_ratio | 预测性能/理论最佳（目标>0.95） |






