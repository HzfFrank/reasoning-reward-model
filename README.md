# SOLE-R1 v3 复现

本分支提供 LIBERO Spatial 上的 v3 数据构造、SFT、GRPO RLVR 和双视角推理。V7 位于本仓库的 `main` 分支，两者分别克隆即可使用。

## 1. 获取代码与环境

```bash
git clone -b v3-reproduction https://github.com/HzfFrank/reasoning-reward-model.git sole-v3
cd sole-v3
```

v3 和 V7 共用 **SOLE-R1 环境**，仿真与策略训练使用 **RLinf 环境**。SOLE-R1 官方在 `reward_server` 下执行 `uv sync`；本项目的安装入口在此基础上补齐共用依赖：

```bash
# 已有 SOLE-R1 环境：直接复用，补齐固定的项目依赖。
bash scripts/install.sh /your/src/sole-r1-model
export PYTHON=/your/src/sole-r1-model/reward_server/.venv/bin/python
export V7_REWARD_PYTHON="$PYTHON"
```

首次使用先按 [SOLE-R1 官方说明](https://github.com/Philip-MIT/sole-r1-model/blob/main/reward_server/README.md) 安装环境，再执行上述适配命令。适配脚本不创建环境或重跑官方安装。如果已从 V7 仓库完成这一步，只需设置 Python 路径。完整安装说明统一见 [SOLE-R1 与 RLinf 环境](https://github.com/HzfFrank/reasoning-reward-model/blob/main/docs/环境_CN.md)。

仿真采集使用 [RLinf 数据采集分支](https://github.com/HzfFrank/RLinf/tree/handoff/v7-data-collection)和同一 RLinf 环境。已有轨迹的 v3 几何计算、CoT 生成、SFT、RLVR 和推理均使用 SOLE-R1 Python。

## 2. 下载现成的数据与模型

| 内容 | Hugging Face |
|---|---|
| 14,046 条 v3 数据及拼图 | [nova819/sole-r1-v3-data](https://huggingface.co/datasets/nova819/sole-r1-v3-data) |
| v3 SFT 模型 | [nova819/sole-r1-v3-sft](https://huggingface.co/nova819/sole-r1-v3-sft) |
| v3 RLVR 模型 | [nova819/sole-r1-v3-rlvr](https://huggingface.co/nova819/sole-r1-v3-rlvr) |

下载数据时须包含 `images/`。训练入口读取 HF `save_to_disk` 格式，可将公开 JSONL 转换一次：

```bash
hf download nova819/sole-r1-v3-data --repo-type dataset --local-dir data/v3
"$PYTHON" data/v3/resolve_paths.py --output data/v3/train.absolute.jsonl
"$PYTHON" -m toolkits.libero_sole.merge_cot_dataset \
  --inputs data/v3/train.absolute.jsonl --output data/v3/training.jsonl \
  --hf-dataset-dir data/v3/arrow
export V3_DATASET="$PWD/data/v3/arrow"
```

SFT 的起点为 `Qwen/Qwen3-VL-8B-Instruct`，原实验固定 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir models/Qwen3-VL-8B-Instruct
export V3_BASE_MODEL="$PWD/models/Qwen3-VL-8B-Instruct"
export V3_RUNS_ROOT="$PWD/runs"
```

已有模型目录直接设置变量即可，不会再次下载。可用 `HF_HOME`、`UV_CACHE_DIR`、`TMPDIR`、`TORCH_EXTENSIONS_DIR` 指定缓存和临时目录。

## 3. 从 rollout 构造数据

原实验使用 404 条成功轨迹。每条包含 `images/main/`、`images/wrist/`、`actions.npy`、`semantic_states.pkl`、`contact_records.pkl`、`task.json` 和元数据。原始轨迹来自 RLinf 采集分支；仅重新训练模型时可直接使用第 2 节的数据。

先准备成功轨迹的工作副本，再运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/build_data.sh /absolute/path/success_trajectories /absolute/path/generated_v3
```

该命令依次计算 contact progress（beta=0.75）、混合抽帧、构造双视角拼图、本地 Qwen 生成 CoT、导出 JSONL 和 Arrow。它会在工作轨迹目录写入分数与拼图；不要将工作副本软链接到需要保持不变的原始轨迹。原版规则保留 30% from-zero 抽样概率，固定随机种子；生成规模随轨迹长度和事件数量变化。

`generate_cot_dataset_v3.py` 调用 v2/v1 中的公共构造函数，因此这些模块都是运行依赖。模板、方向分析及标签规则沿用 v3，未换成 V7 的全局视频协议。

## 4. SFT

```bash
NPROC=4 bash scripts/train_sft.sh
```

默认一轮全数据、全参数 BF16、ZeRO-3、AdamW、学习率 `1e-5`，每卡 microbatch=1、梯度累积16；只监督 assistant completion。输出包括 DeepSpeed checkpoint 和 Hugging Face 模型。改变卡数会改变有效 batch，复现原实验使用4卡。

通过尾部参数可做短运行，例如 `--train-samples 4 --max-optimizer-steps 1 --gradient-accumulation-steps 1`。正式配置在脚本中明确列出。

## 5. RLVR

```bash
export V3_SFT_MODEL=/absolute/path/to/v3_sft_model
NPROC=4 bash scripts/train_rlvr.sh
```

沿用 v3 的 GRPO：500步、group size=8、temperature=0.7、top-p=0.9、KL系数0.01、clip=0.2、tau=5、学习率 `5e-7`。奖励由输出格式和预测百分比误差组成；每50步评估、每100步及结束时导出 HF 模型。该配置复现本地 v3 实验，论文的5000步与 `1e-6` 学习率并非本次运行配置。

RLVR 从 SFT 权重初始化，并保留冻结参考模型；完整状态和训练指标写入 `V3_RUNS_ROOT`。它是 reward model 的训练，不是机器人策略的 PPO 训练。

## 6. 完整轨迹推理

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" inference.py \
  --model /absolute/path/to/v3_rlvr_model \
  --episode /absolute/path/to/episode \
  --mode sequential --stride 4 --output runs/inference/episode.jsonl
```

sequential 使用初始、前一、当前时刻的双视角拼图，并回灌上一次模型预测；from-zero 使用初始和当前双视角，传入 `--mode from_zero`。两种模式共用数据构造的 prompt 函数，推理不读取 progress 标签。输出逐查询保存原文 CoT、解析结果、帧号和实际输入文本。

推理 CLI 使用 Hugging Face 后端，方便在同一训练环境运行；输入拼图与提示词保留 v3 协议。首帧以0%初始化，查询最后一帧，格式错误会保留原文并报错。

## 7. 文件导航

- `toolkits/libero_sole/`：几何分数、抽帧、拼图、CoT 与 Arrow 导出。
- `stage2a_tools/`：正式 SFT 入口与共用 collator。
- `stage3_tools/`：GRPO 入口、训练公共函数、奖励计算与指标记录。
- `scripts/`：安装、数据构造、SFT、RLVR 命令。
- `inference.py`：与训练提示词一致的轨迹推理。

原始实验结果、测试驱动和临时模型留在本地，Git 分支只保存运行代码、配置和说明。

发布前真实运行记录见 [验证记录](docs/验证记录.md)。
