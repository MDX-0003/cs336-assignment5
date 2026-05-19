# GRPO 实验代码审查报告

> 文件：`grpo_experiment.py`
> 对比参考：`sft_experiment.py`（已验证可运行）

---

## 一、致命 BUG（会直接报错或训不了）

### BUG-1：`random.randrange` 参数错误（line 194）

```python
# 当前（错误）：
batch_idxs = [random.randrange(0, len(args.train_samples)) for _ in range(rollout_size)]
```

`args.train_samples` 是 `int`（默认 5），`len(5)` 直接 TypeError。

```python
# 修正：
batch_idxs = [random.randrange(0, len(train_questions)) for _ in range(rollout_size)]
```

---

### BUG-2：训练循环索引完全错误（line 256-267）

当前代码：

```python
for epoch in range(10_000_000):
    for idx in range(train_size):           # train_size=32
        start = idx * train_size            # idx=0: start=0, idx=1: start=32 → OOB
        end = start + train_size            # 每次取 train_size 个样本

        batch_input_ids = input_ids[start:end]       # idx=0 能跑，idx=1 下标越界
        ...
        for micro_batch_idx in range(args.grad_acc_steps):
            mb_input_ids = batch_input_ids[start:end]  # 复用外层 start:end，完全错误！
```

**问题 1**：`idx` 循环 `train_size` 次，每次取 `train_size` 个样本，但 `rollout_size` 总共就 32 个样本（4×8），`idx=1` 时 `start=32, end=64` 直接 IndexError。

**问题 2**：micro batch 循环里又在 `batch_input_ids` 上用 `start:end` 切片，这取出的是整个 batch 而不是 micro batch。

**问题 3**：GRPO 不是 SFT——不应该有 `for epoch` 循环。SFT 有固定数据集需要多 epoch，GRPO **每个 step 都重新 rollout 生成新数据**，它的"batch"是一次性的。当前一个 rollout 只应该产生一个 optimizer step。

**修正**（用一个 optimizer step 处理一次 rollout 的所有样本）：

```python
# GRPO 的"训练"是对一次 rollout 的所有样本做梯度累积
# rollout_size 个样本 → grad_acc_steps 次 micro batch

opt.zero_grad(set_to_none=True)
loss_accum = 0.0
entropies = []

micro_batch_size = rollout_size // args.grad_acc_steps  # 如 32/16=2
for mb_idx in range(args.grad_acc_steps):
    mb_start = mb_idx * micro_batch_size
    mb_end = mb_start + micro_batch_size

    mb_input_ids = input_ids[mb_start:mb_end]
    mb_labels = labels[mb_start:mb_end]
    mb_response_mask = response_mask[mb_start:mb_end]
    mb_advantages = advantages_gpu[mb_start:mb_end]
    mb_old_log_probs = old_log_probs[mb_start:mb_end] if old_log_probs is not None else None

    scored = get_response_log_probs(
        model=policy,
        input_ids=mb_input_ids,
        labels=mb_labels,
        return_token_entropy=True,
    )
    policy_log_probs = scored["log_probs"]
    token_entropy = scored["token_entropy"]

    micro_loss, meta = grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=mb_response_mask,
        gradient_accumulation_steps=args.grad_acc_steps,
        loss_type=loss_type,
        raw_rewards=mb_raw_rewards if loss_type == "no_baseline" else None,
        advantages=mb_advantages if loss_type != "no_baseline" else None,
        old_log_probs=mb_old_log_probs if loss_type == "grpo_clip" else None,
        cliprange=args.cliprange if loss_type == "grpo_clip" else None,
    )
    loss_accum += float(micro_loss.detach().cpu())
    entropy = masked_mean(token_entropy.detach(), mb_response_mask, dim=None)
    entropies.append(float(entropy.cpu()))

grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
opt.step()
opt_step += 1
```

---

### BUG-3：`compute_group_normalized_rewards` 的 `group_size` 参数语义错误（line 217）

```python
# 当前：
advantages, raw_rewards, metadata = compute_group_normalized_rewards(
    ...
    group_size=rollout_size,          # ← 整批 32 个放在一组
    ...
)
```

GRPO 的 group normalization 本意是：**同一个问题被生成 n 次的 n 个 response 放在一组内标准化**（同一问题的不同回答之间比较优劣）。当前把全部 32 个放一组，等于只在 batch 内做标准化——丢失了组内对比信息。

```python
# 修正：
advantages, raw_rewards, metadata = compute_group_normalized_rewards(
    ...
    group_size=args.sample_per_rollout,   # ← 4 个同一问题的回答放一组
    ...
)
```

这样 rollout 的 32 个样本 = 8 个不同问题 × 每个问题 4 个回答 → 8 组，每组内标准化。

---

## 二、缺失的关键阶段

### MISS-1：`log_event` 函数完全缺失

SFT 有嵌套函数 `log_event` 记录训练过程到 `log.jsonl`，GRPO 完全没有。需要添加：

```python
log_path = run_dir / "log.jsonl"
def log_event(event: Dict[str, Any], *, also_print: bool = True):
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "time": time.time(),
        "step": step,
        "opt_step": opt_step,
        **event,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
    if also_print:
        if "msg" in event:
            print(event["msg"])
        else:
            print(payload)
```

同时需要在 `for step in range(args.max_steps)` 之前初始化 `opt_step = 0`。

---

### MISS-2：无训练过程指标记录

当前中间没有任何 loss/reward 记录。需要每 N 步记录一次：

```python
if step % 10 == 0:
    avg_loss = loss_accum / args.grad_acc_steps
    avg_entropy = sum(entropies) / len(entropies)
    log_event({
        "type": "train",
        "loss": avg_loss,
        "entropy": avg_entropy,
        "raw_reward_mean": float(raw_rewards.mean()),
        "advantages_mean": float(advantages.mean()),
    }, also_print=False)
```

---

### MISS-3：无训练结束时的评估（eval）

SFT 在 `step >= args.max_steps` 时触发一次完整 eval。GRPO 完全没有。需要从 sft_experiment.py 照搬 eval 流程：

```python
# step == args.max_steps 时（或通过 eval_interval）：
policy.cpu()
torch.cuda.empty_cache()

# 用基座 vLLM + 当前 policy 权重做评估
eval_llm = init_vllm(
    args.model_path,
    device=args.vllm_device,
    seed=args.seed,
    gpu_memory_utilization=0.50
)
policy.eval()
with torch.no_grad():
    load_policy_into_vllm_instance(policy, eval_llm)
    rows = evaluate_vllm(
        vllm_model=eval_llm,
        reward_fn=r1_zero_reward_fn,
        prompts=val_question,          # 来自 load_jsonl 的结果
        ground_truths=val_gts,
        sample_params=eval_sampling_params,
        request_batch_size=64,
    )
summary = summarize(rows)
log_event({
    "type": "eval_metadata",
    "metadata": summary,
    "msg": f"[step={step}] {summary}"
})

del eval_llm
torch.cuda.empty_cache()
policy.to(args.train_device)
policy.train()
```

需要有独立的 `eval_sampling_params`（和 `rollout_sampling_params` 分开）：

```python
eval_sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=args.max_tokens,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)
```

---

### MISS-4：无训练结束时的模型保存

SFT 在 step 2000 后调用 `policy.save_pretrained()` + `tokenizer.save_pretrained()`。GRPO 需要同样处理：

```python
# 训练循环结束后：
policy.save_pretrained(str(run_dir / "final"))
tokenizer.save_pretrained(str(run_dir / "final"))
log_event({"type": "save", "out_dir": str(run_dir / "final"),
           "msg": f"Saved: {run_dir / 'final'}"})
```

---

### MISS-5：无 `--filter_correct` 的实现

参数定义了但没实现逻辑。如果要从 SFT checkpoint 加载（而非基座模型），应该加：

```python
# 如果传了 --sft_checkpoint，从 SFT 训练好的权重初始化 policy
if args.sft_checkpoint:
    policy = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": args.train_device},
    )
```

但当前的 `--filter_correct` flag 在 GRPO 场景下语义不明——GRPO 用自己的 `train.jsonl` 不需要过滤 SFT 数据。这个参数可能应该删除或改名为 `--load_sft_checkpoint`。

---

### MISS-6：gpu_memory_utilization 设置

当前 `policy_vllm` 初始化用 `gpu_memory_utilization=0.50`，这留了 50% VRAM 给 policy model。对于 RTX 5080 16GB：
- vLLM + Qwen2.5-1.5B 约需 4-5GB
- policy model（训练时）约需 6-8GB（含梯度、optimizer 状态）
- 总计 ~14GB

当前设置合理但紧。如果 OOM，降低到 0.40。

---

## 三、可从 sft_experiment.py 照搬的代码模块

| 模块 | 在 SFT 的位置 | 如何搬到 GRPO |
|------|-------------|--------------|
| `log_event` 嵌套函数 | [sft:176-192](sft_experiment.py) | 直接复制到 `main()` 的 `log_path` 定义之后，把 `step`/`opt_step` 变量初始化提前 |
| `opt_step` 计数器 | [sft:275-276](sft_experiment.py) | `opt_step = 0` 放在 `for step in range(...)` 之前，每次 `opt.step()` 后 `opt_step += 1` |
| 完整 eval 流程（评估 + 清理 + 恢复训练） | [sft:354-406](sft_experiment.py) | 复制 `need_eval` 分支，适配 GRPO 的 `val_question`/`val_gts` |
| `eval_sampling_params` | [sft:279-285](sft_experiment.py) | 复制一份命名 `eval_sampling_params`，区别于 `rollout_sampling_params` |
| 训练结束的 save 逻辑 | [sft:400-402](sft_experiment.py) | 加在 `for step in range(...)` 循环之后 |
| tqdm 进度条后处理 | [sft:329-331](sft_experiment.py) | 复用 `pbar.set_postfix(loss=..., ...)` 在每次 optimizer step 后更新 |

---

## 四、结构性改进建议

### 4.1 统一 step 计数

当前有 `step`（外循环 rollout 次数）和 `global_step`（/`opt_step` optimizer 更新次数），二者在 GRPO 里是 1:1 关系（每个 rollout 产生一次 optimizer step）。直接统一为 `step`：

```python
opt_step = 0
for step in range(1, args.max_steps + 1):
    # ... rollout + train + opt.step() ...
    opt_step += 1
    pbar.update(1)
    pbar.set_postfix(loss=f"{avg_loss:.4f}", reward=f"{float(raw_rewards.mean()):.3f}")
```

### 4.2 random shuffle 可以合并到采样阶段

当前先采样 `batch_idxs`，然后生成 rollout，又用 `torch.randperm` 再 shuffle 一次。两次 shuffle 是冗余的——采样阶段本就是随机采样，生成结果不需要再 shuffle（GRPO 的 group 内顺序对 loss 无影响）。

但 shuffle 在将同组样本分散到 micro batch 时有意义（避免同 prompt 的多个回答集中在同一个 micro batch）。所以保留或删除均可，不影响正确性。

### 4.3 `import random` 已存在但调用方式有问题

`random.randrange(0, len(...))` → `random.randrange(start, stop)` 是合法调用。但前面 BUG-1 已经指出 `len(args.train_samples)` 的问题。

---

## 五、修正后的完整核心训练循环

```python
def main():
    # ... (args 解析、路径、load_jsonl 等保持不变) ...

    # ===== 初始化计数器 & 日志 =====
    opt_step = 0
    run_dir = Path(args.out_path) / f"samples{args.train_samples or 'full'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    def log_event(event: Dict[str, Any], *, also_print: bool = True):
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "time": time.time(),
            "step": step,
            "opt_step": opt_step,
            **event,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
        if also_print:
            if "msg" in event:
                print(event["msg"])
            else:
                print(payload)

    # ===== 加载数据 =====
    prompt_template = Path(args.prompt_path).read_text(encoding="utf-8")
    train_questions, train_gts = load_jsonl(
        args.train_path, prompt_template, args.train_samples, 0
    )
    val_questions, val_gts = load_jsonl(
        args.val_path, prompt_template, args.val_samples, 0
    )

    # ===== 加载模型 =====
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": args.train_device},
    )
    policy.train()

    policy_vllm = init_vllm(
        args.model_path,
        device=args.train_device,
        seed=args.seed,
        gpu_memory_utilization=0.50,
    )
    # 初始时 vLLM 里已是基座模型权重，无需 load_policy_into_vllm_instance
    # 因为 policy_vllm 刚用 model_path 初始化，和 policy 权重相同

    # ===== 优化器 =====
    opt = Adafactor(
        policy.parameters(),
        lr=args.lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    # ===== 采样参数（训练用） =====
    rollout_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # ===== 采样参数（评估用，可以和训练不同） =====
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    rollout_size = args.sample_per_rollout * args.group_size
    micro_batch_size = rollout_size // args.grad_acc_steps

    pbar = tqdm(total=args.max_steps, desc="GRPO training", unit="step")

    for step in range(1, args.max_steps + 1):
        # ========== 1) sample prompts ==========
        batch_idxs = [random.randrange(0, len(train_questions)) for _ in range(rollout_size)]
        batch_prompts = [train_questions[i] for i in batch_idxs]
        batch_gts = [train_gts[i] for i in batch_idxs]

        # ========== 2) policy rollout ==========
        outputs = policy_vllm.generate(batch_prompts, rollout_sampling_params)
        rollout_prompts = []
        rollout_responses = []
        rollout_gts = []
        for request, gt in zip(outputs, batch_gts):
            rollout_prompts.append(request.prompt)
            rollout_responses.append(request.outputs[0].text)
            rollout_gts.append(gt)

        # ========== 3) compute group-normalized advantages ==========
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=rollout_gts,
            group_size=args.sample_per_rollout,       # ← 修正：每组 sample_per_rollout 个回答
            advantage_eps=1e-6,
            normalize_by_std=args.advantage_std,
        )

        # ========== 4) tokenize ==========
        toks = tokenize_prompt_and_output(
            prompt_strs=rollout_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        input_ids = toks["input_ids"].to(args.train_device)
        labels = toks["labels"].to(args.train_device)
        response_mask = toks["response_mask"].to(args.train_device)

        # ========== 5) old policy log probs (grpo_clip) ==========
        old_log_probs = None
        if loss_type == "grpo_clip":
            with torch.inference_mode():
                old_out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
                old_log_probs = old_out["log_probs"].detach()

        # ========== 6) shuffle (打散同 prompt 的多个回答) ==========
        perm = torch.randperm(rollout_size, device=input_ids.device)
        advantages_gpu = advantages.to(args.train_device)[perm].unsqueeze(-1)
        raw_rewards_gpu = raw_rewards.to(args.train_device)[perm].unsqueeze(-1)
        input_ids = input_ids[perm]
        labels = labels[perm]
        response_mask = response_mask[perm]
        if old_log_probs is not None:
            old_log_probs = old_log_probs.to(args.train_device)[perm]

        # ========== 7) 一次 optimizer step：micro batches 梯度累积 ==========
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        entropies = []

        for mb_idx in range(args.grad_acc_steps):
            mb_start = mb_idx * micro_batch_size
            mb_end = mb_start + micro_batch_size

            scored = get_response_log_probs(
                model=policy,
                input_ids=input_ids[mb_start:mb_end],
                labels=labels[mb_start:mb_end],
                return_token_entropy=True,
            )
            micro_loss, meta = grpo_microbatch_train_step(
                policy_log_probs=scored["log_probs"],
                response_mask=response_mask[mb_start:mb_end],
                gradient_accumulation_steps=args.grad_acc_steps,
                loss_type=loss_type,
                raw_rewards=raw_rewards_gpu[mb_start:mb_end] if loss_type == "no_baseline" else None,
                advantages=advantages_gpu[mb_start:mb_end] if loss_type != "no_baseline" else None,
                old_log_probs=old_log_probs[mb_start:mb_end] if loss_type == "grpo_clip" else None,
                cliprange=args.cliprange if loss_type == "grpo_clip" else None,
            )
            loss_accum += float(micro_loss.detach().cpu())
            entropy = masked_mean(scored["token_entropy"].detach(),
                                  response_mask[mb_start:mb_end], dim=None)
            entropies.append(float(entropy.cpu()))

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()
        opt_step += 1

        # 更新 vLLM 中的 policy 权重
        load_policy_into_vllm_instance(policy, policy_vllm)

        # ========== 8) 日志 ==========
        avg_loss = loss_accum / args.grad_acc_steps
        avg_entropy = sum(entropies) / max(len(entropies), 1)
        pbar.update(1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", reward=f"{float(raw_rewards.mean()):.3f}")

        if step % 10 == 0:
            log_event({
                "type": "train",
                "loss": avg_loss,
                "entropy": avg_entropy,
                "grad_norm": float(grad_norm),
                "raw_reward_mean": float(raw_rewards.mean()),
                "raw_reward_std": float(raw_rewards.std()),
            }, also_print=False)

        # ========== 9) eval ==========
        need_eval = (not args.disable_eval
                     and ((args.eval_interval > 0 and step % args.eval_interval == 0)
                          or step == args.max_steps))
        if need_eval:
            policy.cpu()
            torch.cuda.empty_cache()
            eval_llm = init_vllm(
                args.model_path,
                device=args.vllm_device,
                seed=args.seed,
                gpu_memory_utilization=0.50,
            )
            policy.eval()
            with torch.no_grad():
                load_policy_into_vllm_instance(policy, eval_llm)
                rows = evaluate_vllm(
                    vllm_model=eval_llm,
                    reward_fn=r1_zero_reward_fn,
                    prompts=val_questions,
                    ground_truths=val_gts,
                    sample_params=eval_sampling_params,
                    request_batch_size=64,
                )
            summary = summarize(rows)
            log_event({
                "type": "eval_metadata",
                "metadata": summary,
                "msg": f"[step={step}] loss={avg_loss:.4f} {summary}"
            })
            del eval_llm
            torch.cuda.empty_cache()
            policy.to(args.train_device)
            policy.train()

    pbar.close()

    # ===== 最终保存 =====
    final_dir = run_dir / "final"
    policy.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log_event({"type": "save", "out_dir": str(final_dir),
               "msg": f"Saved: {final_dir}"})
```

---

## 六、BUG 总结清单

| # | 严重程度 | 位置 | 问题 |
|---|---------|------|------|
| BUG-1 | 致命 | line 194 | `random.randrange(0, len(args.train_samples))` → `len(int)` TypeError |
| BUG-2 | 致命 | line 256-267 | 训练循环索引：`idx * train_size` 在 idx≥1 时越界；micro batch 切片复用外层 start:end |
| BUG-3 | 重要 | line 217 | `group_size=rollout_size` 应为 `group_size=args.sample_per_rollout` |
| MISS-1 | 致命 | — | 无 `log_event` 函数，无任何日志输出 |
| MISS-2 | 重要 | — | 无训练过程指标记录（loss/reward 不写入 log） |
| MISS-3 | 重要 | — | 无训练结束时评估，val 数据加载了但从未使用 |
| MISS-4 | 重要 | — | 无训练结束时模型保存 |
| MISS-5 | 中等 | — | `--filter_correct` 定义了但无实现 |
| MISS-6 | 中等 | line 192 | `for step in range(args.max_steps)` — step 从 0 开始，和常见 1-based 习惯不一致（不影响功能但日志会差 1） |

---

## 七、可直接照搬 sft_experiment.py 的代码（复制即用）

以下 5 段代码从 `sft_experiment.py` 中提取，可直接粘贴到 `grpo_experiment.py`：

### 7.1 log_event 嵌套函数

```python
log_path = run_dir / "log.jsonl"
def log_event(event: Dict[str, Any], *, also_print: bool = True):
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "time": time.time(),
        "step": step,
        "opt_step": opt_step,
        **event,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
    if also_print:
        if "msg" in event:
            print(event["msg"])
        else:
            print(payload)
```

### 7.2 eval 流程 + 内存管理

```python
policy.cpu()
torch.cuda.empty_cache()
eval_llm = init_vllm(args.model_path, device=args.vllm_device,
                     seed=args.seed, gpu_memory_utilization=0.50)
policy.eval()
with torch.no_grad():
    load_policy_into_vllm_instance(policy, eval_llm)
    rows = evaluate_vllm(eval_llm, r1_zero_reward_fn,
                         val_questions, val_gts,
                         eval_sampling_params, request_batch_size=64)
summary = summarize(rows)
log_event({"type": "eval_metadata", "metadata": summary,
           "msg": f"[step={step}] {summary}"})
del eval_llm
torch.cuda.empty_cache()
policy.to(args.train_device)
policy.train()
```

### 7.3 最终保存

```python
policy.save_pretrained(str(run_dir / "final"))
tokenizer.save_pretrained(str(run_dir / "final"))
log_event({"type": "save", "out_dir": str(run_dir / "final"),
           "msg": f"Saved: {run_dir / 'final'}"})
```

### 7.4 eval_sampling_params（独立的温度参数）

```python
eval_sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=args.max_tokens,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)
```

### 7.5 validate 和 summarize 的使用

`sft_utils.py` 已经提供了 `summarize(rows)` 方法，返回带有 `format_rate`、`answer_accuracy`、`avg_reward`、`counts` 的 dict。评估后直接用它，不需要像 SFT 里手动计算 `eval_acc`/`eval_format`：

```python
summary = summarize(rows)
# summary: {"n": 500, "format_rate": 0.998, "answer_accuracy": 0.696,
#           "avg_reward": 0.696, "counts": {"F1A1": 348, "F1A0": 151, "F0A0": 1}}
```
