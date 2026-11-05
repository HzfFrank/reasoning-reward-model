# 环境与安装边界

环境分三份：reward（v7 数据生成/SFT/推理）、policy（新版 RLinf/Pi0/PPO）、collection（旧版 RLinf/LIBERO，只有重新采集/几何标注需要）。不直接复制旧 `.venv`，不把旧新 RLinf editable 安装进同一环境。

## reward

实测源环境为 Python 3.10.12、torch 2.9.0、transformers 4.57.3。`environments/legacy_reward.observed.txt` 保存所有已安装 distribution 版本；其中包含历史工具，不能盲目把整份当作最小 requirements 或跨平台 lock。

先按 GPU/CUDA 与驱动准备匹配的 PyTorch 2.9.0，再安装 `environments/reward-core.txt`。使用 `scripts/data_disk_env.sh` 保证 pip/HF/Torch 缓存和 tmp 在数据盘。可用 venv/uv 创建独立环境；本轮没有在新机器重建 CUDA 环境，因此不宣称完整安装已验证。

## policy

源环境 Python 3.11.14、torch 2.6.0、transformers 4.53.2、Ray 2.55.1、MuJoCo 3.8.1、robosuite 1.4.1。完整观测版本见 `rl_policy.observed.txt`。

在交付的 RLinf-training 仓库中先阅读 `requirements/install.sh` 与 Pi0/LIBERO 安装说明，选择 embodied、openpi、libero 安装目标；固定所需 OpenPI、LIBERO 源码 revision，不运行不固定版本的全环境升级。旧环境存在 editable LIBERO/OpenPI 依赖，仅有包名和版本号不足以复现，源码 revision 清单仍需补齐后才能称为完整环境锁。

## collection

与 policy 属于近似依赖栈，但使用 RLinf-collection 的旧 env 和 collector 接口。观测版本见 `legacy_rollout.observed.txt`。采集代码未来合入新 RL 分支后再考虑合并环境，目前不强行合并。

## 重要

HF 模型权重、Pi0 norm_stats、paligemma tokenizer、LIBERO BDDL/assets/initial states 是额外运行资产，不随 pip 安装自动齐全。RL 在离线模式读取它们，缺失应在启动前解决。

本轮又确认 policy 环境需要 `environments/policy-extra.txt` 中的附加包。原实验通过 compat_deps 目录加载它们，交接环境应安装到自己的 policy 环境。LIBERO 的实际 commit 和是否有 tracked 修改记录在 `environments/source_dependencies.json`。
