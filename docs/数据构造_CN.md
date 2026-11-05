# 数据构造与固定模板

优先直接使用已发布正式数据。重新生成需要原始 rollout：main/wrist PNG、动作、完整仿真状态、semantic_states、接触记录、任务对象身份和 reset 信息。HF 的训练图像包不是完整仿真轨迹备份。

## 成功 CoT

源 v3 14,046 条，404 个成功 episode；保留既有 split。设置 `V7_V3_SOURCE`、`V7_ROLLOUTS`、`V7_BASE_MODEL`；split 默认读仓库的 `provenance/success_episode_splits.json`。

```bash
"$V7_REWARD_PYTHON" stage7_tools/v7_bulk_grounded.py --output-dir "$V7_RUNS_ROOT/success_cot_01" --scan-only
"$V7_REWARD_PYTHON" stage7_tools/v7_bulk_grounded.py --output-dir "$V7_RUNS_ROOT/success_cot_01" --local-check
# 仅主动执行下一条才批量调用 API：
"$V7_REWARD_PYTHON" stage7_tools/v7_bulk_grounded.py --output-dir "$V7_RUNS_ROOT/success_cot_01" --run --workers 4
```

四条已认可样例作为本地模板回归夹具，不参与训练扩增。原生成器保留源行数、split、协议和断点校验。模型固定 gpt-6.1-sol，API 由 `V7_API_BASE_URL`、`V7_API_KEY_FILE` 配置，SDK 不自动重试；调用记录与用量保存在输出目录，不纳入发布源码。

## 失败 CoT

失败只构造 sequential，使用已确认失败窗口的冻结几何标签。模板是 `v7_short_failure_teacher.py`，继承 `v7_spatial_def_teacher.py` 与正式成功模板。保留同样的 XML 渲染、22 图输入和 student prompt；teacher 的仿真对象字典/状态不能泄漏进 student 输入。

已有候选 JSONL 和原始轨迹可用以下通用入口（替代依赖六条 pilot 输出的老批量主函数）：

```bash
"$V7_REWARD_PYTHON" stage7_tools/generate_failure_cot.py \
  --candidates /your/data/failure_candidates.jsonl \
  --success-snapshot /your/data/success \
  --tokenizer "$V7_BASE_MODEL" --output "$V7_RUNS_ROOT/failure_cot_01"
```

默认只准备、校验，不提交请求。主动加 `--generate --max-requests 6` 才提交最多六条；之后可增加明确预算。目录有排他锁，已保存结果复用；存在请求记录但没有结果时不自动重发，以免重复计费。候选路径必须已映射到本机，`semantic_states.pkl` 只读取可信的自采轨迹文件。

## 几何标注、切窗与采集

正式 v2 使用同一两距离定义，beta=0.75；每个 task 使用自己的冻结 train 成功锚点。没有按失败视频终点强制归一化到100%。核心实现 `production_failure_score.py`、`geometry_progress_v2.py`，事件对齐在 `sample_failure_event_aligned.py`，最终筛选在 `finalize_failure_event_aligned.py`。

这些阶段的历史调度仍读取固定实验 manifest。源码和依赖已经保留，但原始轨迹、按任务校准清单和选窗输入尚未整理成公开独立资产包，不能声称新机器已能一键重新采集并重建116条失败数据。已完成的训练数据可独立用于 SFT。这部分待交接配置继续完善；不要重新校准公式来掩盖缺少历史输入。

旧采集器及 collector 改动在独立 RLinf 采集分支，暂不合入当前 RL 训练分支。日后合并时以新 RL 环境为目标，保留观测/动作对应与完整状态记录。

采集与几何复算前，设置 `V7_COLLECTION_REPO` 为 RLinf 采集分支的克隆目录，并将其加入 `PYTHONPATH`；使用该分支的 policy/仿真环境。该变量不能指向 RL 训练分支。原始轨迹中的 BDDL/图像路径也需映射到本机。
