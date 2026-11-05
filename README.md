# V7 progress reward model

这是 v7 正式流程的源码交接仓库：成功/失败数据构造、固定 CoT 模板、全参数 SFT、完整轨迹推理，以及与独立 RLinf 仓库的接入说明。模型、训练图像和 JSONL 放在 Hugging Face；实验输出不纳入 Git。

本仓库从成熟实验代码按依赖整理，保留 `stage7_sft` 和 `stage7_tools` 名称，避免模块重命名破坏现有协议。少量名称带 review/pilot/v1 的文件是正式入口仍引用的公共实现，不代表需要重新执行历史试验。

## 先读哪些文件

- [使用说明](docs/使用说明_CN.md)：下载数据后的 SFT、推理与 RL 连接。
- [环境说明](docs/环境_CN.md)：reward、policy、collection 三套环境及已知安装边界。
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

上述是发布目标名称，使用前核对上传是否完整；本代码整理不代表后台权重上传已经完成。

RL 使用另一个保留上游历史的 RLinf 仓库。reward 为 `clip(progress,0,100)/100 - 1`；完整轨迹 sequential stride 20、线性插值、每副本 batch 4。协议失败按 RL 接口规则排除整条 episode；不要用伪造分数填补。
