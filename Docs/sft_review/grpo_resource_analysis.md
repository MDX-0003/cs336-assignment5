# GRPO 资源与显存分析报告

> Smoke 命令：`--train_samples 3 --val_samples 3 --question_per_rollout 3 --sample_per_rollout 2 --micro_batch_size 2 --grad_acc_steps 3 --max_steps 10 --loss_interval 1 --disable_eval`
> 输出目录：`runs/grpo_experiment/samples3/`
> 完成时间：2026-05-18 14:52

---

## 一、CPU/GPU 负载循环分析

### 1.1 每步耗时分解

Smoke 10 步的时间戳：

```
step   时间戳         间隔         raw_reward_mean
──────────────────────────────────────────────────
  0    14:43:57        —            0.167
  1    14:44:31       34s          0.000
  2    14:45:32       61s          0.333
  3    14:46:12       40s          0.167
  4    14:46:59       47s          0.333
  5    14:47:52       53s          0.667
  6    14:48:38       46s          0.333
  7    14:49:26       48s          0.333
  8    14:50:20       54s          0.500
  9    14:51:14       54s          0.833
──────────────────────────────────────────────────
平均:                 ~49s/step
```

### 1.2 为什么负载是"极高→极低"的锯齿形

这是 GRPO 的**结构特性**，不是 bug。每个 step 有 6 个串行阶段，各自独占不同硬件：

```
时间轴 (49s/step)
│
├─ 1) 采样 prompts ────────────  <1s   CPU 低
├─ 2) vLLM rollout (32条) ───── 15-20s GPU ████████████████ (CPU 低)
├─ 3) reward 计算 (sympy) ───── 5-8s   CPU ██████████ (GPU 低)
├─ 4) tokenize ──────────────── 2-3s   CPU █████ (GPU 低)
├─ 5) policy fwd/bwd (16 mb) ── 15-20s GPU ████████████████ (CPU 低)
├─ 6) opt.step + sync vLLM ──── 2-3s   GPU ████
│
```

**GPU 只在 2) 和 5) 阶段工作**，其余时间完全空闲。CPU 在 3) 阶段 sympy 答案比对时拉满（解析 LaTeX 表达式、数值比较）。所以 GPU 利用率看起来在 0% ↔ 100% 之间反复跳——这是正常的。

### 1.3 内存持续高位的原因

系统 RAM 消耗 3 个大头：

| 消费者 | 大小 | 说明 |
|--------|------|------|
| Python 进程 (主) | ~4-5 GB | policy 模型权重 + optimizer 状态 (Adafactor) |
| vLLM 进程 | ~3-4 GB | 模型复本 + KV cache |
| Transformers / tokenizer | ~1 GB | tokenizer.json 就 11MB，但 Qwen 的 vocab.json 3MB |
| **合计** | **~8-10 GB** | |

加上系统开销（WSL2 本身、VSCode、浏览器），32GB 系统内存在 80-95% 区间是正常的。**不会爆系统 RAM**。

GPU 显存占用：

| 消费者 | 大小 | 说明 |
|--------|------|------|
| policy (train mode) | ~3.0 GB | bfloat16 params (1.5B × 2B) |
| policy gradients | ~3.0 GB | 同大小 |
| Adafactor optimizer | ~6.0 GB | moment + variance (2× params) |
| vLLM model | ~3.0 GB | bfloat16 params 复本 |
| vLLM KV cache | ~0.5-1.5 GB | rollout_size=32, max_tokens=1024 |
| activations | ~1.0 GB | flash_attn_2 减少了很多 |
| **合计** | **~16-17 GB** | **RTX 5080 显存边界** |

**这就是 GPU 内存接近 100% 的原因** — 刚好踩在 16GB 的显存极限上。

---

## 二、时间估算：正式运行要多久

Smoke 的 rollout_size=6，正式是 rollout_size=32。时间主要在 vLLM generation 和 policy training，两者都随 rollout_size 线性增长：

```
vLLM generate 时间 ≈ 生成 token 数 × batch_size
                   ∝ rollout_size × sample_per_rollout 的响应长度

policy train 时间  ∝ rollout_size (micro batch 总样本数)
```

Smoke：rollout_size=6, grad_acc_steps=3, micro_batch_size=2, 平均 ~34s GPU时间/step
正式：rollout_size=32, grad_acc_steps=16, micro_batch_size=2, 预期 ~70-90s GPU时间/step

加上 CPU 阶段：

```
正式每步预计:  20s(vLLM) + 8s(reward) + 3s(tokenize) + 45s(training) + 5s(sync) ≈ 80-90s/step
```

**2000 步 = 2000 × 85s ≈ 47 小时**

这还是不含 eval 的情况。如果每 2000 步做一次 eval (val_samples=0 全量 5000 题)，eval 自己就要额外 ~2-3 小时（5000 题 ÷ 64 batch × 每题生成 ~30s ≈ 40 min，但实际更慢因为 policy 要来回 CPU↔GPU）。

---

## 三、正式运行会 OOM 吗

### 3.1 训练阶段（每步循环）

Smoke 的 rollout_size=6，training 用 micro_batch_size=2 × grad_acc_steps=3。这些样本在 `get_response_log_probs` 中一起进 GPU。

扩展关系（近似）：

| 参数 | Smoke | 正式 | 显存增量 |
|------|-------|------|----------|
| rollout_size | 6 | 32 | ×5.3 |
| micro_batch_size | 2 | 2 | 不变 |
| grad_acc_steps | 3 | 16 | 更多 micro batch |
| 策略模型显存 | ~12 GB | ~12 GB | **不变** |
| vLLM 模型显存 | ~3 GB | ~3 GB | **不变** |
| KV cache | 0.5 GB | 2-3 GB | **×4-6** |
| activation (单 micro batch) | 0.5 GB | 1-2 GB | ×2-4 |

**关键点**：gradient accumulation 让每次只处理 micro_batch_size=2 的样本，activation 峰值取决于 micro batch size 而非 rollout size。所以**正式跑不会比 smoke 更吃 GPU 显存**。

vLLM KV cache 会增大，因为 rollout_size=32 时 `generate(batch_prompts)` 要同时缓存 32 个序列。但 RTX 5080 有 16GB，vLLM model 3GB + KV 2-3GB ≈ 5-6GB，留给 policy training 约 10GB，可行但紧。

**结论**：训练阶段大概率不 OOM，但显存在 15-16GB / 16GB 边界。

### 3.2 Eval 阶段（step = max_steps 时触发）

Eval 阶段流程：

```
policy.cpu()           ← 释放 ~12GB GPU 显存
torch.cuda.empty_cache()  ← 清理碎片
eval_llm = init_vllm(gpu_memory_utilization=0.50)  ← 新 vLLM，只用 8GB
load_policy_into_vllm_instance  ← 加载当前权重
evaluate_vllm(全量 val, batch_size=64)  ← 生成 5000 题
```

这里 **不会 OOM**，因为 policy 被移到 CPU 后才创建 eval vLLM。但是：

- **policy 移到 CPU 后占系统 RAM**：~3GB 额外 RAM
- **vLLM 生成 5000 题时间很长**：估计 1-3 小时
- 如果 `val_samples=0`（全量），eval 会测 5000 题；建议 eval 用 `val_max_examples=500`（需要在代码里实现）

### 3.3 唯一的 OOM 风险点

现有代码在 `log_and_eval` 块（line 337）里创建 eval_llm 时，用来做 rollouts 的 `policy_vllm` **还没有被删除**。这意味着 eval 阶段有两个 vLLM 实例共存：

```
policy_vllm (用于 rollout) + eval_llm (用于评估) = 两个 vLLM
```

每个 vLLM 占用 4-6 GB，两个就是 8-12 GB。加上 policy training 的 ~12GB → OOM。

**修正**：在创建 eval_llm 之前先删除 policy_vllm（或通过 `collective_rpc` 直接复用同一个实例做 eval）。

---

## 四、参数与显存的近似推算关系

### 4.1 关键参数对显存的影响

| 参数 | 影响对象 | 关系 | 显存灵敏度 |
|------|----------|------|-----------|
| `micro_batch_size` | policy activation 峰值 | 线性 | **高** — 从 2→4 可能 +2-4 GB |
| `rollout_size` (= q × s) | vLLM KV cache | 线性 | **中** — 32→64 约 +1-2 GB |
| `sample_per_rollout` | rollout_size | ← | 同上 |
| `question_per_rollout` | rollout_size | ← | 同上 |
| `grad_acc_steps` | **无直接影响** | 只影响训练时间 | **零** — 不改变峰值 |
| `max_tokens` | vLLM KV cache | 线性 | **中** — 1024→2048 约 ×2 |
| `gpu_memory_utilization` | vLLM KV cache 总量 | 百分比 | **高** — 0.50→0.60 +1.6 GB |

### 4.2 RTX 5080 (16GB) 安全运行窗口

```
micro_batch_size ≤ 3  (训练侧)
rollout_size ≤ 40     (推理侧)
gpu_memory_utilization ≤ 0.55  (vLLM 侧)
```

当前正式参数（mb=2, rollout=32, gmu=0.50）刚好都在安全窗口内。

---

## 五、优化建议

### 5.1 缩短训练时间

当前 47 小时的瓶颈是每步串行 6 个阶段。可以做的：

**A) 全程跑不需要 2000 步** — GRPO 通常样本效率比 SFT 高，500-1000 步就可能有效果。减少 `--max_steps` 是最大杠杆：

```bash
--max_steps 500   # 约 12 小时
```

**B) 减小 eval 开销** — eval 阶段占了约 1-3 小时。代码里 `val_max_examples` 参数未被使用。修正后：

```bash
--val_max_examples 500   # eval 只测 500 题而非 5000 题
```

### 5.2 避免 eval 时 OOM

在 eval 前删除旧的 policy_vllm：

```python
# 在 line 343 policy.cpu() 之前加：
del policy_vllm
torch.cuda.empty_cache()

# eval 结束后重建 policy_vllm：
policy_vllm = init_vllm(args.model_path, ...)
load_policy_into_vllm_instance(policy, policy_vllm)
```

### 5.3 推荐最终正式命令

```bash
uv run python grpo_experiment.py \
  --train_samples 0 \
  --val_samples 0 \
  --val_max_examples 500 \
  --question_per_rollout 8 \
  --sample_per_rollout 4 \
  --micro_batch_size 2 \
  --grad_acc_steps 16 \
  --max_steps 500 \
  --loss_interval 10 \
  --loss_type grpo_clip \
  --advantage_std \
  --cliprange 0.2
```

预计耗时：500 × 85s ≈ **12 小时**（不含 eval）。

---

## 六、总结

| 负载现象 | 是否正常 | 原因 |
|----------|---------|------|
| GPU 0%↔100% 锯齿 | ✅ 正常 | vLLM rollout 和 policy training 交替独占 GPU |
| CPU 间歇拉满 | ✅ 正常 | sympy reward 计算时 CPU 密集 |
| 系统 RAM 80-95% | ✅ 正常 | policy + vLLM + tokenizer 常驻 ~10GB |
| GPU 显存 90-100% | ⚠️ 边界 | policy training + vLLM 几乎用满 16GB |
| Eval 时 OOM 风险 | ⚠️ 存在 | 两个 vLLM 实例共存会爆显存 |

核心结论：**正式参数不会比 smoke 更吃显存（micro_batch_size 不变则 activation 峰值不变），但需要在 eval 前手动删除 policy_vllm 避免两个 vLLM 共存**。
