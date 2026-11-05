# 环境安装与使用

日常使用只需要两个环境：reward（CoT构造、Qwen SFT和推理）与policy（RLinf、Pi0和LIBERO）。旧扰动采集/几何标注已用同一个policy环境完成小规模真实验证，无需默认安装第三套环境。旧、新RLinf代码仍为两个分支，运行时分别选择源码；不要同时editable安装两份RLinf。

以下入口面向Linux x86_64/NVIDIA。先安装uv，并准备兼容的NVIDIA驱动。目录可使用系统盘、用户目录或共享数据盘；本项目不强制存储位置。需要自定义缓存时，先编辑并source `configs/paths.example.sh`，再source `scripts/data_disk_env.sh`。

## Reward环境

在v7仓库根目录执行：

```bash
bash scripts/install_reward.sh "$HOME/envs/v7-reward"
export V7_REWARD_PYTHON="$HOME/envs/v7-reward/bin/python"
```

脚本创建独立Python 3.10环境，安装Torch 2.9.0/CUDA 12.8、torchvision 0.24.0和Transformers 4.57.3等直接依赖，再检查依赖一致性。`reward-linux-py310.constraints.txt`固定参考环境的传递依赖，不是适用于任意平台的通用锁。已有目录会被拒绝，不自动覆盖旧环境。

## Policy环境

先检出交接记录对应的RLinf训练分支；在v7仓库根目录运行：

```bash
bash scripts/install_policy.sh /your/src/RLinf /your/envs/rlinf-policy
export V7_POLICY_PYTHON=/your/envs/rlinf-policy/bin/python
export LIBERO_CONFIG_PATH=/your/envs/rlinf-policy/libero_config
```

该入口先进入RLinf仓库根目录，调用它的`requirements/install.sh embodied --model openpi --env libero`，再应用`policy-compat.txt`集中记录的兼容版本并运行`uv pip check`。系统依赖已经满足且不允许sudo时，在命令末尾加`--no-root`；否则由官方脚本安装系统依赖。参考路线使用SDPA，跳过FlashAttention编译。

这里不复制旧venv，也不使用历史`compat_deps`。三个曾经手动补充的包lz4、zstandard、sortedcontainers已经是交接版RLinf正式依赖。

### 为什么还有一个兼容文件

官方多阶段安装会改变前一阶段安装的依赖。实际从空环境安装发现五项声明冲突，必须显式收敛：

- rlinf-openpi 0.1.1要求Torch 2.7.1，不能继续沿用旧环境的2.6.0；配套torchvision/torchaudio/torchcodec一并固定。
- dm-control的新版本要求MuJoCo>=3.8.1，但rlinf-libero 0.1.3要求<3.4。固定dm-control 1.0.34与MuJoCo 3.3.7。
- OpenPI专用`rlinf-transformer-openpi`与通用`transformers`提供同名Python模块。两份包元数据都固定为4.53.2，并最后重装OpenPI补丁包，避免通用包覆盖其模型实现。
- Ray固定为此前平台使用的2.55.1，不随解析器升级到新版本。

这些处理全部在安装入口中执行，不要求协作者自行猜版本或复制本机补丁。修改版本时先运行依赖检查及最小真实验证。

## 两个环境如何连接

启动前设置`V7_REWARD_PYTHON`、`V7_POLICY_PYTHON`、两个仓库位置以及模型/输出路径，示例在`configs/paths.example.sh`。RL入口使用policy Python启动训练，reward worker使用reward Python加载模型；不需要在终端反复activate两个环境。

Pi0权重及norm_stats、Qwen权重及processor、paligemma tokenizer是单独资产，不能用pip安装代替。官方安装器下载LIBERO assets。多套LIBERO环境共存时使用不同`LIBERO_CONFIG_PATH`，避免缓存配置指向另一套环境。

## 旧采集环境与验证边界

旧采集分支可复用上述policy Python。安装仍以RLinf训练分支为入口；运行采集/几何标注时，在独立进程中显式选择采集源码：

```bash
export V7_COLLECTION_REPO=/your/src/RLinf-collection
export PYTHONPATH="$V7_COLLECTION_REPO:$V7_REWARD_REPO/stage7_tools"
# 使用 $V7_POLICY_PYTHON 运行采集或几何标注入口。
# 启动RL训练前退出此shell，或unset PYTHONPATH，由训练入口选择训练源码。
```

不要在同一Python进程中切换两份RLinf；模块缓存不会随PYTHONPATH自动替换。两分支使用同一环境的LIBERO_CONFIG_PATH即可。

实测：task6成功基线127动作、配对掉落186动作；前缀完整状态误差0，313帧几何标注无恢复异常；另复算历史task7的240帧，与旧环境两距离/progress完全一致。未验证所有任务与扰动，也不保证新旧环境重新rollout逐帧相同。旧环境版本清单保留为历史复现参考，不再作为协作者必装项。采集脚本的历史输出目录限制和原始资产交接仍须单独处理，不能把环境兼容当作一键重建全部数据。

实际安装/运行验证状态见[交接状态](交接状态_CN.md)。本机Ubuntu 22.04已有系统图形库和驱动，独立新建Python环境不等于重装操作系统；两卡测试也不等于八卡平台验收。
