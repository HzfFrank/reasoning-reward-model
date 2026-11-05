# 交接状态：2026-11-05

本地源码整理版；尚未声称 GitHub 发布或异机全流程验收完成。

完成：独立 v7 仓库结构；必要源码依赖闭包；正式 teacher/student 模板原文保留；数据盘配置；成功/混合八卡训练入口；固定曝光计划重建；完整轨迹推理入口；失败 CoT 通用预检/生成入口；RL 独立分支可配置部署路径；旧采集修改归档分支。

实测：从发布版成功和失败数据重建14,704条曝光计划；混训预检通过，919步、460/919保存点不变。RL reward CPU回归11项通过（需要原 compat_deps）；真实 Hydra 配置路径解析通过，global batch2048、rollout epoch8、max epochs1000不变。一条冻结失败候选的通用入口预检通过，API调用为0。

环境仍须注意：observed清单不是跨平台安装锁。LIBERO源码revision已记录；OpenPI及GPU/CUDA安装仍需按RLinf实际依赖完成异机验证。没有复制旧venv，也没有新建或升级本机环境。

数据构造的剩余边界：几何标注/切窗的历史调度依赖原始轨迹、状态与实验manifest；本仓库保留实现及选窗/标尺摘要，但没有把这些原始资产作为源码提交，也未宣称新机器可以一键重建原始失败轨迹。CoT通用入口要求已准备并映射路径的候选及其semantic_states。

没有运行GPU训练、rollout或付费API。现有实验代码、数据、模型、上传任务均未被替换。本地所有检查产物放在仓库外的checks目录。

补充实测：成功sequential、成功from_zero、失败sequential三个正式样本通过真实Qwen processor，图像数22/20/22、prompt labels全部-100、student输入逐项一致；成功版/完整版模型的prompt文件哈希与协议校验均通过。没有加载GPU权重。
