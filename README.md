# V7 progress reward model

这是 v7 正式流程的源码交接仓库。包含成功/失败数据构造、固定 CoT 模板、全参数 SFT、完整轨迹推理，以及与独立 RLinf 仓库的接入说明。模型、训练图像和 JSONL 放在 Hugging Face。

## 说明

- [使用说明](docs/使用说明_CN.md)：下载数据后的 SFT、推理与 RL 连接。
- [环境说明](docs/环境_CN.md)：SOLE-R1、RLinf 两套环境，以及旧采集分支共用 policy 的方法。
- [协作规则](CONTRIBUTING.md)：分支、兼容版本、实验记录。
- [数据构造](docs/数据构造_CN.md)：模板、原始 rollout、几何分数和 API 入口。
- [交接状态](docs/交接状态_CN.md)：已验证项与仍需完成的可移植工作。

## 固定协议

sequential 输入 GLOBAL 16 图、INITIAL 2 图、PREVIOUS 2 图、CURRENT 2 图和上一进度；from_zero 输入 GLOBAL 16 图、INITIAL 2 图、CURRENT 2 图。GLOBAL 是完整 episode 均匀抽取的 8 个时刻，每个时刻 main/wrist 双视角。推理依赖完整轨迹，不是因果在线单帧评分。

输出固定为 `<think><update>…</update><state>…</state><progress>…</progress></think><answer>整数%</answer>`。student 模板在 `stage7_sft/prompts.py`；成功 teacher 在 `stage7_tools/v7_grounded_contract_templates.py`；失败 teacher 在 `stage7_tools/v7_short_failure_teacher.py`，继承关系已保留。

成功数据 train 13,364、validation 336、test 343；失败数据仅 train 116。完整版训练覆盖全部成功 train 一次，按冻结计划加入 1,336 次失败曝光和 4 条尾部 padding，总计 14,704 次曝光、八卡有效 batch 16、919 次 optimizer 更新。不是 14,704 个独立样本。

## 发布资产

- 模型：`nova819/sole-r1-v7-success`、`nova819/sole-r1-v7-full`
- 数据：`nova819/sole-r1-v7-success-data`、`nova819/sole-r1-v7-failure-data`

## 配套代码仓库

- 本仓库：[HzfFrank/reasoning-reward-model](https://github.com/HzfFrank/reasoning-reward-model)。克隆目录可自定，统一用 `V7_REWARD_REPO` 指向它。
- RL训练：[HzfFrank/RLinf 的 handoff/v7-rl-training 分支](https://github.com/HzfFrank/RLinf/tree/handoff/v7-rl-training)。
- 旧数据采集：[同仓库的 handoff/v7-data-collection 分支](https://github.com/HzfFrank/RLinf/tree/handoff/v7-data-collection)。仅需重建原始轨迹时克隆到另一目录。

安装 policy 环境使用 RL 训练分支作为安装入口；采集分支可复用该环境，不必再装第三套。

V3 与 V7 共用 SOLE-R1 的 `reward_server/.venv`；RLinf 的 V7 评分进程也使用该 Python。安装与复用命令见上面的环境说明。
