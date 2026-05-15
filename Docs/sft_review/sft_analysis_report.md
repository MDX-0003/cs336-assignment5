# SFT 实验分析报告

> 实验命令：`uv run python sft_experiment.py`
> 输出目录：`runs/sft_experiment/samples500_all/`
> 完成时间：2026-05-15 18:11

---

## 一、`runs/` 目录文件来源

### 1.1 目录命名规则

代码 [sft_experiment.py:244](sft_experiment.py#L244)：

```python
run_dir = Path(args.out_path) / f"samples{args.train_samples or 'full'}_{'filtered' if args.filter_correct else 'all'}"
```

本次运行参数均为默认值：`--out_path runs/sft_experiment`、`--train_samples 500`、未传 `--filter_correct`，因此路径为 `samples500_all`。"all" 表示训练数据**未按答案正确性过滤**，全部 sft.jsonl 样本都参与了训练。

### 1.2 文件清单

目录共 11 个文件：

| 文件 | 来源 | 说明 |
|------|------|------|
| `model.safetensors` | [line 400](sft_experiment.py#L400)：`policy.save_pretrained()` | 训练后 Qwen2.5-Math-1.5B 的完整权重 |
| `config.json` | 同上 | 模型结构定义（qwen2, 28层, 1536 hidden, 12 heads, 1.5B 参数） |
| `generation_config.json` | 同上 | 推理时的默认采样参数 |
| `tokenizer.json` | [line 401](sft_experiment.py#L401)：`tokenizer.save_pretrained()` | BPE 分词器模型文件 |
| `vocab.json` | 同上 | 词表 (~152K tokens) |
| `merges.txt` | 同上 | BPE merge rules |
| `tokenizer_config.json` | 同上 | 分词器配置（tokenizer class、special tokens 等） |
| `special_tokens_map.json` | 同上 | 特殊 token 映射 |
| `added_tokens.json` | 同上 | 额外添加的 token |
| `chat_template.jinja` | 同上 | Qwen2.5 用的对话模板 |
| `log.jsonl` | [line 255](sft_experiment.py#L255)：`log_event()` | 训练过程日志（每 10 个 opt_step 记录一条） |

**核心理解**：`policy.save_pretrained()` 保存的不是"增量"（LoRA adapter 之类），而是**完整模型权重**。这意味着这个目录本身就是一个可直接加载的 HuggingFace 模型：

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("runs/sft_experiment/samples500_all")
```

---

## 二、训练过程还原

### 2.1 三次运行，两次失败

log.jsonl 共 38 行（36 条 train_loss + 1 条 eval_metadata + 1 条 save），按时间戳可识别出 **3 次独立运行**：

| 运行 | 时间 | step 范围 | 末次 loss | 结局 |
|------|------|-----------|-----------|------|
| Run 1 | 17:01:53 → 17:05:23 | 160 → 1920 | 0.563 | ❌ 在 eval 阶段崩溃 |
| Run 2 | 17:08:58 → 17:12:28 | 160 → 1920 | 0.602 | ❌ 同上 |
| Run 3 | 18:06:18 → 18:11:33 | 160 → 2000 | **0.535** | ✅ 完成 |

**为什么前两次在 step 1920 中断？** step=1920 时 `opt_step=120`，下一步 step=2000 会触发 eval（代码 [line 348-352](sft_experiment.py#L348-L352)）：

```python
need_eval = (not args.disable_eval and (
    (args.eval_interval > 0 and step % args.eval_interval == 0)
    or step == args.max_steps  # ← 2000 % 1 == 0 或直接命中 max_steps
))
```

eval 阶段需要初始化 vLLM 实例，vllm 0.7.2 → 0.10.0 的 API 不兼容正是在此处暴露。修复后才有了 Run 3 的成功。

### 2.2 关键超参

从代码默认参数可知：

| 参数 | 值 | 含义 |
|------|-----|------|
| `micro_batch_size` | 2 | 每步处理的样本数 |
| `grad_acc_steps` | 16 | 梯度累积步数 → 有效 batch size = 2×16 = 32 |
| `max_steps` | 2000 | 总训练步数 → opt_step = 2000/16 = 125 次权重更新 |
| `lr` | 2e-5 | Adafactor 学习率 |
| `train_samples` | 500 | 从 sft.jsonl 中随机抽取 500 条训练 |
| 完整 sft.jsonl | 1767 条 → 用了 28.3% | — |

### 2.3 Run 3 的 Loss 曲线

```
opt_step          loss
─────────────────────────
  10 (step 160)   1.625  ████████████████
  20 (step 320)   1.266  █████████████
  30 (step 480)   0.871  █████████
  40 (step 640)   1.031  ██████████
  50 (step 800)   0.543  █████
  60 (step 960)   0.656  ██████
  70 (step 1120)  0.941  █████████
  80 (step 1280)  0.434  ████
  90 (step 1440)  0.820  ████████
 100 (step 1600)  0.320  ███
 110 (step 1760)  0.210  ██
 120 (step 1920)  0.205  ██
 125 (step 2000)  0.535  █████  ← eval 中记录
─────────────────────────
```

**解读**：

- Loss 从 1.6 级别降到 0.2 级别，整体下降趋势明确
- 震荡幅度较大（0.21 → 0.94 之间波动），原因是 `micro_batch_size=2` 导致**单步梯度噪声很高**，梯度累积 16 步相当于平滑了但未完全消除
- opt_step=125（eval 时）loss=0.535 > opt_step=120 的 0.205，不是模型变差了，而是 eval 过程中的 loss 来自不同 batch

---

## 三、评估结果深入分析

### 3.1 最终指标

```
[step=2000] loss=0.5352
  eval/accuracy:      0.572   ← 57.2% 答案正确
  eval/format_rate:   0.798   ← 79.8% 回复格式正确
  eval/avg_reward:    0.572   
  eval/n:             500     ← 验证 500 题
```

### 3.2 Reward 函数如何打分

核心逻辑在 [cs336_alignment/drgrpo_grader.py:1008-1047](cs336_alignment/drgrpo_grader.py#L1008-L1047)，流程图：

```
                      ┌─────────────────────────┐
                      │   模型输出 response       │
                      └───────────┬─────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │ 包含 "</think> <answer>" 且            │
              │ 包含 "</answer>" ?                     │
              └───────────────────┬───────────────────┘
                        Yes │                    │ No
              ┌─────────────┴──────┐     ┌───────┴─────────────┐
              │ 提取 <answer> 标签  │     │ format_reward = 0   │
              │ 中间的内容         │     │ answer_reward = 0   │
              └─────────┬───────────┘     │ reward = 0          │
                        │                  └─────────────────────┘
              ┌─────────┴────────────────────────┐
              │ grade(model_answer, ground_truth) │
              │ 用 math_verify + sympy 比较答案   │
              └─────────┬────────────────────────┘
                 正确 │                    │ 错误
          ┌──────────┴──────┐     ┌────────┴──────────────────┐
          │ format_reward=1 │     │ format_reward=1           │
          │ answer_reward=1 │     │ answer_reward=0           │
          │ reward=1         │     │ reward=0                  │
          └─────────────────┘     └───────────────────────────┘
```

**Reward 是二值的**：只有 1（格式对 + 答案对）或 0（其他）。不存在中间分数。

### 3.3 avg_reward = accuracy 意味着什么

`avg_reward = 0.572 = accuracy` 这说明：**所有格式正确的回复中，答案正确率恰好 = 整体 accuracy**。数学关系推导：

设 500 个样本中：
- `F` = format_reward=1 的数量 = 500 × 0.798 = **399**
- `A` = answer_reward=1 的数量 = 500 × 0.572 = **286**

avg_reward = (只有 F=1 且 A=1 的样本才能得 reward=1) / 500

由于 `reward == answer_reward`（格式对了但答案错 reward 也是 0），所以 avg_reward 永远等于 accuracy。

**这意味着什么？** format_rate 和 accuracy 是解耦的两个维度。你可以单独提升任一个而不影响另一个。当前：

- 399 人"看懂了答题格式"（输出 `</think> <answer> ... </answer>`）
- 其中 286 人还答对了
- 113 人格式对了但答错了（reward=0）
- 101 人连格式都没对（reward=0）

### 3.4 format_rate 为什么只有 79.8%？—— 深入分析

模型必须**精确**输出 `</think> <answer> 答案 </answer>` 结构。不像多选题有明确的格式提示，数学答案的格式要求对模型来说是隐含的——它需要从 prompt 中学会：

```
<think> reasoning process here </think> <answer> answer here </answer>
```

**为什么 20% 的回复格式不对？** 可能的原因：

1. **训练数据质量**：sft.jsonl 中的 response 不一定都严格遵循 `</think> <answer> ... </answer>` 格式。如果部分样本格式不规范，SFT 会学到错误的格式习惯。
2. **SFT 样本量有限**：500 个样本、125 步优化，模型可能尚未充分内化格式规则。
3. **temperature=1.0 的高随机性**：eval 时 `temperature=1.0`，采样过程有较大随机性，模型可能"走偏"生成不完整格式。
4. **基座模型倾向**：Qwen2.5-Math-1.5B 预训练时习惯自由格式输出，SFT 未完全覆盖这一倾向。

**验证方法**：检查 sft.jsonl 中 response 的格式合规率。如果训练数据格式规范率是 X%，模型学到的是 X%，那么 eval format_rate 通常 ≤ X%。

---

## 四、Score 计算方法（关键补充）

代码 [sft_experiment.py:377-380](sft_experiment.py#L377-L380)：

```python
n = len(rows)
eval_acc  = sum(r.answer_reward for r in rows) / n   # 答案正确率
eval_format = sum(r.format_reward for r in rows) / n   # 格式合规率
eval_reward = sum(r.reward for r in rows) / n          # 综合 reward（= accuracy）
```

因为 `reward = format_reward AND answer_reward`（两者同时为 1 时 reward 才是 1），所以：

```
eval_reward = accuracy = 0.572
```

**这里有一个关键点**：`reward` 不是 `format_reward + answer_reward`，而是**逻辑与**关系。也就是说，模型拿不到"半对"的分数。格式对了但答案错 = 0 分。这导致 reward = accuracy。

---

## 五、提升方案

### 5.1 优先提升 accuracy（57.2% → 目标 70%+）

**方案 A：过滤训练数据（最简单，预计 +5-10%）**

```bash
uv run python sft_experiment.py --filter_correct
```

代码 [line 65-88](sft_experiment.py#L65-L88) 的 `filter_correct_sft_samples` 会筛选出**答案正确的样本**，只对"格式对 + 答案对"的样本进行 SFT。当前 1767 条 sft.jsonl 中答案正确的比例约 57% → 过滤后约剩 1000 条，质量更高。

**方案 B：增大训练数据量**

```bash
uv run python sft_experiment.py --train_samples 0  # 0 = 全量 1767 条
```

500 条是 1767 条的 28%，模型见过不够多样式。全量训练需要约 3 倍时间（1767/32 ≈ 55 batch × 3 epoch = 165 opt_steps）。

**方案 C：两个方案叠加**

```bash
uv run python sft_experiment.py --filter_correct --train_samples 0
```

先过滤出正确答案样本，再用全部正确样本训练。预计效果最好。

### 5.2 提升 format_rate（79.8% → 目标 95%+）

**方案 A：检查训练数据格式**

检查 sft.jsonl 中的 response 是否都遵循 `</think> <answer> ... </answer>`：

```bash
uv run python -c "
import json
total = ok = 0
with open('data/MATH/sft.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        resp = ex.get('response', '')
        total += 1
        if '</think> <answer>' in resp and '</answer>' in resp:
            ok += 1
print(f'格式合规率: {ok}/{total} = {ok/total:.1%}')
"
```

如果训练数据本身格式合规率不高，需要先修复数据质量。

**方案 B：降低 eval temperature**

在 [sft_experiment.py:279-285](sft_experiment.py#L279-L285) 中 `eval_sampling_params` 的 `temperature=1.0` 不变，但可以改命令行：

不过 eval 阶段 temperature=1.0 是固定写死的。如果要改，编辑 `eval_sampling_params` 把 temperature 降到 0.7。

**方案 C：增加 `--max_steps`**

当前只训 2000 步。增加到 4000-6000 步让模型有更多机会内化格式。

### 5.3 稳定训练（减少 loss 震荡）

增大有效 batch size 来平滑梯度：

```bash
uv run python sft_experiment.py --grad_acc_steps 32 --max_steps 4000
```

grad_acc_steps=32 → 有效 batch = 2×32 = 64，梯度噪声更低。

### 5.4 推荐起步命令

按优先级排列：

```bash
# 第一步：先看训练数据格式质量
uv run python -c "..."

# 第二步：数据过滤 + 全量训练（预计 accuracy 提升到 65%+）
uv run python sft_experiment.py --filter_correct --train_samples 0

# 第三步：如果还不够，加大训练步数
uv run python sft_experiment.py --filter_correct --train_samples 0 --max_steps 4000
```

---

## 六、总结

| 维度 | 当前状态 | 核心问题 |
|------|----------|----------|
| 模型 | Qwen2.5-Math-1.5B → SFT 后 | 1.5B 参数，容量有限 |
| 数据 | 500/1767 条 sft.jsonl | 只用了一小部分，未过滤 |
| 训练 | 125 opt_steps, loss 1.6→0.2 | 在收敛，但 batch 太小导致震荡 |
| accuracy | 57.2% | SFT 模仿样本行为，未做 RL 优化 |
| format_rate | 79.8% | 1/5 回复格式不对 |

**下一步核心动作**：用 `--filter_correct` 过滤训练数据 + `--train_samples 0` 用全量正确样本，预计 accuracy 可提升 5-10 个百分点。


1. SFT 一般训多少？
和预训练不一样，SFT 数据量小、目标单一（模仿格式+答案），3-5 个 epoch 通常就够了。太多 epoch 会导致过拟合——模型背样本而不是学到解题能力。

2. 和数据量什么关系？
算一下就知道了：


有效 batch size = micro_batch_size × grad_acc_steps = 2 × 16 = 32

完整 sft.jsonl:                 1767 条 → 1767/32 ≈ 55 opt_steps/epoch
过滤后（预估 60-70% 正确）:     ~1100 条 → 1100/32 ≈ 34 opt_steps/epoch
当前 500 条:                     500 条 → 500/32 ≈ 16 opt_steps/epoch
当前 max_steps=2000 → opt_steps=125：


当前 500 条:    125/16 ≈ 7.8 epoch    ← 偏多
全量 1767 条:   125/55 ≈ 2.3 epoch    ← 刚好
过滤后 ~1100:   125/34 ≈ 3.7 epoch    ← 刚好
结论：

用 --train_samples 500（当前）：125 opt_steps ÷ 16 ≈ 8 epoch，已经偏多了
用 --train_samples 0 --filter_correct（过滤全量）：125 opt_steps ÷ 34 ≈ 3.7 epoch，合理
用 --train_samples 0（不过滤全量）：125 opt_steps ÷ 55 ≈ 2.3 epoch，略少
所以 max_steps 不需要增。 过滤全量训练时 2000 步能跑 ~4 epoch，对 SFT 来说是标准配置。如果你追求 best result，可以稍微加到 --max_steps 3000（约 5.5 epoch），但提升空间不大。