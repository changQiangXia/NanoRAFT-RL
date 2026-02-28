# NanoRAFT-RL

基于 RAFT 思路的检索增强微调与 PPO 对齐实验工程（Qwen2-7B + LoRA + 4bit）。

## 实验背景

本项目面向“检索增强监督微调 + 强化学习对齐”的闭环训练问题，目标是在单卡 24GB 显存条件下稳定复现以下链路：

- 检索增强样本（含干扰上下文）驱动的 SFT 训练
- SFT 模型上的 PPO 对齐训练
- 基线/SFT/PPO 三模型统一评估

实验重点不只是分数提升，还包括工程可复现性：

- 首次模型下载后可复用本地缓存
- 在网络波动和镜像切换场景下仍能稳定执行
- 对 LoRA adapter 产物进行一致化评估

## 实验设置（2026-02-28 实测）

在当前环境（RTX 4090 24GB、Python 3.10、torch 2.1.2+cu121）下，以下链路已实测跑通：

- SFT：`outputs/raft-sft/final`
- PPO：`outputs/raft-ppo/final`
- 评估：`outputs/evaluation_results.json`

评估配置为 `data/synthetic/test.jsonl`，实际样本数为 10（`--max_samples 30` 但测试集仅 10 条）。

## 实验结果评估

本次实测评分如下：

| 模型 | 格式得分 | CoT得分 | 答案得分 | 综合得分 |
|---|---:|---:|---:|---:|
| 基线模型(Base) | 0.000 | 0.560 | 0.640 | 0.360 |
| SFT模型 | 0.850 | 0.960 | 0.760 | 0.856 |
| PPO模型 | 0.800 | 0.870 | 0.660 | 0.779 |

结果解读：

- SFT 相对基线综合得分提升显著（`0.360 -> 0.856`），主要体现在格式与 CoT 结构学习。
- PPO 在保持较高结构分的同时，综合得分稳定高于基线（`0.779 > 0.360`）。
- SFT 与 PPO 的相对表现会受采样随机性影响，属于生成式模型评估常见现象。

## 创新性（方法与工程实现）

1. 检索增强训练样本的噪声鲁棒设计
- 训练样本包含干扰上下文（distractors），强化“从噪声中定位关键信息”的能力。

2. 单卡 24GB 的可执行训练闭环
- 在 4bit + LoRA + 显存控制策略下，完成 Qwen2-7B 的 SFT 与 PPO 全流程。

3. 可复现导向的工程改造
- 统一脚本缓存目录到 `models/cache`，实现“首次下载、后续复用”。
- 评估脚本兼容 LoRA adapter 目录加载（自动读取 `adapter_config.json`）。
- PPO 训练循环改为严格执行 `total_steps`，并加入全局随机种子控制。

---

## 关键复现问题说明

### 1) 模型加载问题怎么解决？首次与后续有什么区别？

核心策略：统一 Hugging Face 缓存目录到项目内 `models/cache`。

目前脚本行为：

- `scripts/run_raft_sft.py`：默认使用 `models/cache`
- `scripts/run_ppo_training.py`：默认使用 `models/cache`
- `scripts/run_evaluation.py`：默认使用 `models/cache`
- `scripts/run_data_synthesis.py`：默认使用 `models/cache`

这意味着：

- 首次运行：会下载缺失模型文件到 `models/cache`
- 后续运行：优先复用本地缓存，不再重复下载

如需后续严格离线（完全不访问外网），可额外加：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 2) `data.zip` 解压位置如何设置？

是，但要注意目录层级。

`/root/autodl-tmp/data.zip` 的顶层是 `processed/`、`synthetic/` 等，不带 `data/` 前缀。
因此正确命令是：

```bash
cd /root/autodl-tmp/NanoRAFT-RL
unzip -o /root/autodl-tmp/data.zip -d data
```

解压后应出现：

- `data/processed/index/...`
- `data/synthetic/train.jsonl`
- `data/synthetic/val.jsonl`
- `data/synthetic/test.jsonl`

若直接 `-d .`，会变成 `processed/` 落在仓库根目录，路径不匹配脚本默认配置。

### 3) 现在能否保证 GitHub 拉下来后完整复现（拿到数据后）？

谨慎结论：

- 可以高置信复现：`SFT -> PPO -> Eval`（前提：拿到 `data.zip`，并有 24GB 显存）
- 不能 100% 保证跨机器“完全一致数值”：生成式训练存在随机性
- 不能 100% 保证“全流程含数据重建”稳定：数据重建依赖外部 API 与网络（Zhipu/HF）

换句话说：

- 对“工程可跑通并产出模型与评估文件”这一目标：可以保证（在同类环境）
- 对“任何机器、任何时刻、所有分数逐位一致”：不能保证

---

## 推荐复现路径（拿到 data.zip 后）

### 0. 环境

```bash
cd /root/autodl-tmp/NanoRAFT-RL
source /etc/network_turbo
pip install -r requirements.txt
```

### 1. 数据就位

```bash
unzip -o /root/autodl-tmp/data.zip -d data
```

### 2. SFT

```bash
python -u scripts/run_raft_sft.py --config configs/raft_sft.yaml
```

### 3. PPO（可完成版配置）

```bash
python -u scripts/run_ppo_training.py --config configs/ppo_rl_repro.yaml
```

说明：当前实现会严格按照 `total_steps` 执行（dataloader 耗尽后自动续跑）。

### 4. 评估

```bash
python -u scripts/run_evaluation.py \
  --test_file data/synthetic/test.jsonl \
  --max_samples 30 \
  --base_model Qwen/Qwen2-7B-Instruct \
  --seed 42
```

---

## 严格“从原始文档全量重建”路径（更慢、更不稳定）

这条路径会重新做数据合成，依赖外部 API 与网络，不建议在验收阶段首选。

```bash
cd /root/autodl-tmp/NanoRAFT-RL
source /etc/network_turbo
pip install -r requirements.txt

export ZHIPU_API_KEY="<YOUR_KEY>"
python -u scripts/run_data_synthesis.py --config configs/data_synthesis.yaml --force-regenerate

python -u scripts/run_raft_sft.py --config configs/raft_sft.yaml
python -u scripts/run_ppo_training.py --config configs/ppo_rl_4090.yaml
python -u scripts/run_evaluation.py --test_file data/synthetic/test.jsonl --max_samples 30 --base_model Qwen/Qwen2-7B-Instruct --seed 42
```

---

## 缓存机制说明（首次下载后复用）

默认缓存目录：

- `models/cache`

建议保留该目录，避免每次重拉 7B 分片。常用检查：

```bash
du -sh models/cache
find models/cache -maxdepth 3 -type f | head
```

如需验证“只用本地缓存”：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 关键输出

- SFT模型：`outputs/raft-sft/final`
- PPO模型：`outputs/raft-ppo/final`
- 评估结果：`outputs/evaluation_results.json`
- 运行日志：`logs/`

---

## 常见问题（简版）

1. 模型又开始下载，感觉没走缓存
- 检查是否在项目目录运行；确认 `models/cache` 存在并有权重分片。

2. `data.zip` 解压后脚本找不到数据
- 大概率解压层级错了。必须是 `.../NanoRAFT-RL/data/processed`，不是 `.../NanoRAFT-RL/processed`。

3. 如何保证多次跑结果更稳定
- SFT 默认 `seed=42`；PPO 与评估可通过配置/参数固定种子（`seed=42`）。
- 仍需注意：生成模型在不同驱动/算子实现下可能有微小差异，属于正常范围。

4. 网络抖动导致 HF 或 API 失败
- 先执行 `source /etc/network_turbo`；必要时切换 `HF_ENDPOINT` 到官方或镜像。

---

## 变更与踩坑记录

完整错误与修复记录见：

- `reproduction_error_log.md`
