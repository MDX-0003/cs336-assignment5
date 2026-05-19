# GRPO 实验环境配置报告（V100 32GB）

## 核心结论

在 V100 (compute capability 7.0) 上成功运行 smoke 测试的关键：

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| vllm 0.7.2 tokenizer 报错 | `Qwen2Tokenizer` 无 `all_special_tokens_extended` | 升级到 vllm 0.11.2 |
| V100 不支持 bf16 | CC 7.0 < 8.0 | 改用 fp16 |
| V100 不支持 FA2 | CC 7.0 < 8.0 | 改用 sdpa |
| V1 引擎多进程 OOM | vllm 子进程单独加载一份模型 | `VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| KV cache 过大挤占训练显存 | 默认 0.35 * 32GB ≈ 11GB | 降到 0.15 ≈ 4.8GB |
| flash_attn ABI 不兼容 | PyPI wheel 不匹配 torch 2.9 | 用项目本地 wheel |

## 1. 运行时环境变量

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0   # 关键！vllm 跑在主进程内，不额外占一份模型显存
VLLM_ATTENTION_BACKEND=TRITON_ATTN # V100 可用的 attention 后端
```

## 2. 依赖版本

| 包名 | 原版本 | 最终版本 |
|------|--------|----------|
| `vllm` | `0.7.2` | `0.11.2` |
| `torch` | 未锁定 | `2.9.0` |
| `transformers` | `5.8.1` | `4.57.6` |
| `flash-attn` | PyPI | 本地 wheel for torch 2.9 |

## 3. pyproject.toml 关键修改

```diff
- "torch",
+ "torch==2.9.0",

- "vllm==0.7.2",
+ "vllm==0.11.2",

[tool.uv.sources]
+ flash-attn = { path = "flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" }
```

## 4. grpo_experiment.py 修改

| 修改点 | 原值 | 新值 | 原因 |
|--------|------|------|------|
| monkey-patch | `patch("vllm.worker.worker.Worker...")` | 移除 | vllm 0.11.x 不再需要 |
| `attn_implementation` | `"flash_attention_2"` | `"sdpa"` | V100 不支持 FA2 |
| 所有 dtype | `torch.bfloat16` | `torch.float16` | V100 不支持 bf16 |
| `gpu_memory_utilization` | `0.35` | `0.15` | 为 policy 训练留出更多显存 |

## 5. 完整运行命令

```bash
# Smoke 测试
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  uv run python grpo_experiment.py @grpo_smoke.args

# 完整运行（500步，每100步eval）
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  uv run python grpo_experiment.py @grpo_full.args
```

## 6. Smoke 测试结果

```
3/3 [03:03, 61.25s/step, loss=0.0070, opt_step=3]
Saved: runs/grpo_smoke_test/samples8
```

每步约 61 秒（首步包含 CUDA graph 编译缓存，后续步骤会更快）。
