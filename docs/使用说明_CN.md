# 使用说明

先按[环境说明](环境_CN.md)适配已安装的 SOLE-R1 环境，并准备 RLinf 环境。所有 `/your/...` 路径须替换为本机位置。先复制 `configs/paths.example.sh` 到 Git 之外的私有配置文件，填写路径后 source，再执行 `source scripts/data_disk_env.sh`。目录可放在任意可写磁盘；默认使用用户目录，也可设置到共享存储。

## 已发布数据训练

下载完整数据仓库，包括 images。success 和 failure 分开放置在 `V7_DATA_ROOT` 下。使用对应数据仓库的 `resolve_paths.py` 将相对图片引用转换成本机绝对路径；原始 JSONL 不覆盖。

成功版：

```bash
bash scripts/train_success.sh /your/data/success/train.local.jsonl "$V7_BASE_MODEL" "$V7_RUNS_ROOT/success_01"
```

完整版严格重建原曝光顺序，输入可直接使用 HF 相对路径 JSONL：

```bash
"$V7_REWARD_PYTHON" scripts/prepare_frozen_mix.py \
  --success /your/data/success/train.jsonl \
  --failure /your/data/failure/train.jsonl \
  --model "$V7_BASE_MODEL" --output "$V7_RUNS_ROOT/mixed_01"
bash scripts/train_mixed.sh "$V7_RUNS_ROOT/mixed_01"
```

正式入口默认使用单机八卡 torchrun，先做 CPU 预检，再启动 GPU 训练。完整版入口也支持 `NPROC=2/4`，通过梯度累积维持有效 batch=16。模型初始化是 Qwen3-VL-8B-Instruct，不是从成功版 reward checkpoint 继续训练。完整版准备工具只支持已发布的 13,364+116 独立样本，严格使用冻结 exposure plan。扩充数据后须明确建立新计划，不能悄悄套用旧计数。

最小真实混训检查可使用独立准备目录执行：

```bash
NPROC=2 bash scripts/train_mixed.sh "$V7_RUNS_ROOT/mixed_check" --max-optimizer-steps 2 --stop-after-step 1
NPROC=2 bash scripts/train_mixed.sh "$V7_RUNS_ROOT/mixed_check" --max-optimizer-steps 2
```

第一条保存第1步后退出，第二条自动恢复优化器、调度器、RNG与数据位置，完成第2步并导出 `final_model`。先用上面的准备工具建立 `mixed_check`；不要与正式运行共用目录。完整 checkpoint 含模型和优化器，单份约130 GiB，两个 checkpoint 加导出模型建议预留至少320 GiB配额。默认正式预算仍为919次更新。

## 完整轨迹推理

需要原始 rollout 双视角 PNG 与该轨迹的 `task.json`；训练 JSONL 的局部图片集合不保证能重建完整视频。使用冻结模型目录中的全部 tokenizer/processor/协议文件。

```bash
"$V7_REWARD_PYTHON" stage7_sft/optimized_final_episode_inference.py \
  --model-dir "$V7_REWARD_MODEL" \
  --episode-dir /your/data/rollouts/EPISODE \
  --task-file /your/data/rollouts/EPISODE/task.json \
  --output-dir "$V7_RUNS_ROOT/inference_01" \
  --mode sequential --strides 4,20 --device cuda:0
```

from_zero 使用相同命令替换 mode。线性 patch 优化保持原权重，但 BF16 算子差异可能导致少量生成差异。这里是单轨迹离线评测；跨轨迹 batch4 评分由 RLinf reward worker 实现。

## RLinf 接入

另克隆/检出 RLinf 的 `handoff/v7-rl-training` 分支。RLinf Python 负责策略与仿真，`V7_REWARD_PYTHON` 指向共用的 SOLE-R1 Python；入口自动启动评分子进程。

除 paths 示例外设置：

```bash
export V7_POLICY_MODEL=/your/data/models/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
export V7_ARTIFACTS="$V7_DATA_ROOT/runs/rlinf"
export V7_TOKENIZER_ASSET=/your/data/assets/paligemma_tokenizer.model
```

还需按 RLinf/OpenPI 安装说明备齐 LIBERO 资产及 Pi0 norm_stats。离线模式不会自动下载这些资产。

```bash
bash "$V7_RLINF_TRAINING/local_integration/v7_reward/native/run_portable_campaign.sh" \
  short --campaign "$V7_ARTIFACTS/campaigns/experiment_01"
# 短训证明满足本 campaign 的要求后：
bash "$V7_RLINF_TRAINING/local_integration/v7_reward/native/run_portable_campaign.sh" \
  long --campaign "$V7_ARTIFACTS/campaigns/experiment_01"
```

短长训规则沿用 RL 分支实现。本次在新建 RLinf 环境和共用 SOLE-R1 环境中完成了原生 Ray/FSDP 两卡采集、评分、PPO、保存及独立重启续训，并检查了 TensorBoard 指标；正式八卡仍使用平台短训入口。
