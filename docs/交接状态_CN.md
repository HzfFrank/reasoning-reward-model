# 交接状态：2026-11-05

本地源码整理版，尚未推送 GitHub。正式 teacher/student 模板保持原文；实验输出、模型、临时验收驱动和旧测试不纳入源码。

## 已完成的真实小规模验证

以下均运行发布目录中的代码，使用本机已有环境，未覆盖原实验目录。

| 阶段 | 实际验证 |
| --- | --- |
| 几何标注 | 使用 RLinf 采集分支恢复一条真实轨迹的240帧仿真状态，按冻结task标尺复算；所选窗口两个端点与冻结整数标签一致，无恢复异常 |
| CoT构造 | 固定 gpt-6.1-sol 模板真实调用1次，成功导出1条sequential训练JSONL；真实loader/processor验证22图及completion-only监督 |
| 全参数SFT | 2张A100，已有正式成功from_zero和失败sequential各1条，从已有完整版模型继续1次更新；8,767,123,696参数可训练，loss=0.028077；视觉与语言权重实际改变 |
| 保存与推理 | 保存checkpoint、optimizer、scheduler、step和processor；导出8个权重分片；RL reward worker实际重载本轮新模型并生成合法输出 |
| RL接入 | 发布RLinf代码从SFT Pi0真实采集task6的17动作/18观测time-limit轨迹；sequential stride20查询0和17，线性插值回填17个奖励；4个microbatch、1次原生PPO更新，动作输出层32,768个权重值改变 |

SFT小测使用两条已有正式训练样本；本轮新API样本另行通过真实loader/processor检查，未声称它参与了上述参数更新。RL使用的是上述小测新导出的reward模型，不是跳过SFT环节直接使用旧模型。reward保持冻结，公式及输入输出协议未变。

**验证边界：**SFT实测两卡FSDP；RL实测为仓库外最小驱动调用发布版真实环境、reward进程、GAE和PPO方法，未在本轮重新运行原生Ray/FSDP八卡campaign。没有评估学习收益，也没有重建一套全新机器环境。不能将该结果当作八卡平台或异机安装验收。

本机证据保存在发布仓库之外的 `checks/real_pipeline_20261105`；历史检查另包括14,704条曝光计划重建、正式八卡训练预检和RL配置解析。源码精简前后的归档保存在仓库外 `local_archive`，不需要协作者下载历史测试或中间模型。

## 仍需交接的资产边界

已发布训练图像和JSONL可直接用于SFT。重新采集及几何标注还需要原始rollout、仿真状态、接触记录、BDDL/assets、任务标尺和选窗manifest；训练图像包不等于原始轨迹包。历史选窗调度仍依赖实验manifest，不能声称新机器可一键重建116条原始失败样本。

采集与几何复算须明确设置 `V7_COLLECTION_REPO`，指向独立RLinf采集分支，避免隐式导入本机旧目录。RL训练使用独立训练分支。reward/policy/collection环境与安装边界见环境文档；observed版本清单不是跨平台安装锁。
