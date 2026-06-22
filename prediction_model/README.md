# Rho 预测模型

本目录实现了 `rho_prediction_model_design.txt` 中设计的模型，用于从 `tau` 和
`rewards` 预测 `rho` 的条件预测分布参数。

## 环境

请使用项目指定的 `Crowd` conda 环境。推荐直接使用环境里的 Python：

```powershell
D:\anaconda3\envs\Crowd\python.exe
```

以下命令默认在 `prediction_model` 目录运行：

```powershell
cd D:\Workspace\Project\research-docs\prediction_model
```

## 数据准备

原始数据文件为：

```text
dataset/sumo_ryl_initial_population.pkl
```

该 pickle 文件很大，并且包含经验库、策略等训练 `rho` 预测模型不需要的字段。
第一次运行前需要先转换成轻量 `.pt` 缓存：

```powershell
D:\anaconda3\envs\Crowd\python.exe prepare_dataset.py
```

默认输出：

```text
dataset/sumo_ryl_prediction_dataset.pt
```

注意：第一次转换会扫描约 10GB 的 `sumo_ryl_initial_population.pkl`，可能耗时较长。
转换完成后，训练和测试都只读取轻量 `.pt` 缓存。

## 训练

```powershell
D:\anaconda3\envs\Crowd\python.exe train.py --epochs 200 --batch-size 8
```

默认会读取：

```text
dataset/sumo_ryl_prediction_dataset.pt
```

默认输出训练结果到：

```text
runs/rho_prediction/
```

其中最重要的文件是：

- `best.pt`：验证集 NLL 最优 checkpoint。
- `last.pt`：最后一个 epoch 的 checkpoint。
- `history.csv`：每个 epoch 的训练和验证指标。
- `config.json`：训练参数、数据划分和标准化参数。

训练默认会显示每个 epoch 的 train/valid 进度条。如果不想显示，可以加：

```powershell
D:\anaconda3\envs\Crowd\python.exe train.py --no-progress
```

如果显存不足，可以减小 batch size：

```powershell
D:\anaconda3\envs\Crowd\python.exe train.py --epochs 200 --batch-size 2
```

如果 `--batch-size 1` 仍然爆显存，说明单个样本里的车辆数已经太大。
这时应减小每次送入 Road Transformer 的车辆块大小，并开启混合精度：

```powershell
D:\anaconda3\envs\Crowd\python.exe train.py `
  --epochs 200 `
  --batch-size 1 `
  --vehicle-chunk-size 32 `
  --amp
```

仍然不够时继续减小：

```powershell
D:\anaconda3\envs\Crowd\python.exe train.py `
  --epochs 200 `
  --batch-size 1 `
  --vehicle-chunk-size 16 `
  --hidden-dim 64 `
  --road-layers 2 `
  --checkpoint-road `
  --amp
```

如果想先快速确认流程能跑通，可以用较少 epoch：

```powershell
D:\anaconda3\envs\Crowd\python.exe train.py --epochs 2 --batch-size 4
```

## 测试

```powershell
D:\anaconda3\envs\Crowd\python.exe test.py --split test
```

默认会读取：

```text
runs/rho_prediction/best.pt
```

默认输出测试结果到：

```text
outputs/rho_prediction/
```

主要输出：

- `test_metrics.json`：NLL、MAE、RMSE、90% prediction interval coverage 等指标。
- `test_predictions.csv`：每个测试样本的真实 `rho`、预测 `rho`、90% 区间、`sigma`、`nu`。

也可以测试其他 split：

```powershell
D:\anaconda3\envs\Crowd\python.exe test.py --split valid
D:\anaconda3\envs\Crowd\python.exe test.py --split train
D:\anaconda3\envs\Crowd\python.exe test.py --split all
```

## 烟测

如果只是想确认代码环境和脚本是否正常，可以先用一个很小的合成数据集跑 smoke test。
下面命令会生成一个临时合成数据集：

```powershell
@'
import torch
from pathlib import Path
Path("dataset").mkdir(parents=True, exist_ok=True)
n, n_vehicle, n_road = 12, 9, 7
tau = torch.randn(n, n_vehicle, n_road, 2)
rewards = torch.randn(n, n_vehicle, n_road)
rho = tau[..., 0].mean(dim=(1, 2)) + 0.5 * rewards.mean(dim=(1, 2))
torch.save(
    {"tau": tau, "rewards": rewards, "rho": rho, "groups": list(range(n))},
    "dataset/synthetic_prediction_dataset.pt",
)
'@ | D:\anaconda3\envs\Crowd\python.exe -
```

然后训练和测试：

```powershell
D:\anaconda3\envs\Crowd\python.exe train.py `
  --data dataset/synthetic_prediction_dataset.pt `
  --output-dir runs/smoke `
  --epochs 1 `
  --batch-size 4 `
  --hidden-dim 32 `
  --road-layers 1 `
  --device cpu

D:\anaconda3\envs\Crowd\python.exe test.py `
  --data dataset/synthetic_prediction_dataset.pt `
  --checkpoint runs/smoke/best.pt `
  --output-dir outputs/smoke `
  --split test `
  --device cpu
```

## 代码结构

- `model.py`：Road Transformer 编码器、DeepSets 车辆集合聚合、统计特征分支、Student-t 预测头。
- `data.py`：轻量数据集读取、只基于训练集拟合标准化、数据划分、batch 拼接、统计特征。
- `prepare_dataset.py`：把 `sumo_ryl_initial_population.pkl` 转换为 `sumo_ryl_prediction_dataset.pt`。
- `export_dataset_csv.py`：把 `sumo_ryl_prediction_dataset.pt` 导出为 CSV。
- `train.py`：基于 Student-t NLL 训练，支持 early stopping 和 checkpoint。
- `test.py`：输出 NLL、MAE、RMSE、coverage 以及预测 CSV。

## 导出数据集 CSV

默认导出每个 individual 一行的汇总 CSV：

```powershell
D:\anaconda3\envs\Crowd\python.exe export_dataset_csv.py
```

默认输出：

```text
outputs/dataset_csv/sumo_ryl_prediction_summary.csv
```

如果要导出完整的 vehicle-road 明细，行数会非常大，建议压缩：

```powershell
D:\anaconda3\envs\Crowd\python.exe export_dataset_csv.py --format long --long-gzip
```

快速预览前 10 个样本：

```powershell
D:\anaconda3\envs\Crowd\python.exe export_dataset_csv.py --max-samples 10
```

## 诊断常数预测

如果测试结果里 `rho_pred` 几乎是常数，先跑一个小样本过拟合实验：

```powershell
D:\anaconda3\envs\Crowd\python.exe train.py `
  --epochs 100 `
  --batch-size 1 `
  --overfit-samples 16 `
  --vehicle-chunk-size 16 `
  --hidden-dim 64 `
  --road-layers 2 `
  --checkpoint-road `
  --amp `
  --lr 1e-4
```

判断方式：

- 如果 `valid_mae` 能快速降到很低，说明模型能记住样本，常数预测更可能是泛化信号弱或 split 问题。
- 如果小样本也无法下降，说明模型结构、输入特征或训练目标仍有问题。

正式训练时可以保留均值辅助损失，避免 Student-t NLL 只学到“均值 + 不确定性”：

```powershell
D:\anaconda3\envs\Crowd\python.exe train.py `
  --epochs 200 `
  --batch-size 1 `
  --vehicle-chunk-size 16 `
  --hidden-dim 64 `
  --road-layers 2 `
  --checkpoint-road `
  --amp `
  --lr 1e-4 `
  --mean-loss-weight 1.0
```

## 从仓库根目录运行

如果你在仓库根目录 `D:\Workspace\Project\research-docs`，也可以使用模块方式运行：

```powershell
D:\anaconda3\envs\Crowd\python.exe -m prediction_model.prepare_dataset `
  --input prediction_model/dataset/sumo_ryl_initial_population.pkl `
  --output prediction_model/dataset/sumo_ryl_prediction_dataset.pt

D:\anaconda3\envs\Crowd\python.exe -m prediction_model.train `
  --data prediction_model/dataset/sumo_ryl_prediction_dataset.pt `
  --output-dir prediction_model/runs/rho_prediction

D:\anaconda3\envs\Crowd\python.exe -m prediction_model.test `
  --data prediction_model/dataset/sumo_ryl_prediction_dataset.pt `
  --checkpoint prediction_model/runs/rho_prediction/best.pt `
  --output-dir prediction_model/outputs/rho_prediction
```
