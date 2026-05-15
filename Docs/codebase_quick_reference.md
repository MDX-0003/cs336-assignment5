# CS336 Assignment 5 代码速查表

这份速查表的目标是帮你先建立一个“全局地图”。

这份作业的一个重要特点是：

- `cs336_alignment/` 目前几乎没有现成实现
- 真正最值得先读的是 `tests/` 和 `tests/adapters.py`
- 也就是说，这个仓库更像是“测试先给你，功能要你自己补”

所以你在理解代码时，最好的阅读顺序不是“从源码读到测试”，而是反过来：

1. 先看 `tests/test_sft.py`
2. 再看 `tests/test_grpo.py`
3. 再看 `tests/adapters.py`
4. 最后自己在 `cs336_alignment/` 中放实现

---

## 1. 仓库整体结构

### 关键目录

- `cs336_alignment/`
  你真正写代码的地方。目前基本是空的，意味着你有较大自由度自行组织实现。

- `tests/`
  这是目前最重要的“已存在逻辑”。它定义了作业要求、输入输出格式和正确性标准。

- `tests/_snapshots/`
  这里保存了很多 `.npz` 快照文件。测试会把你的输出和这些参考结果比较。

- `tests/fixtures/`
  这里放了测试用的小模型、tokenizer 文件、样例数据等。

- `cs336_alignment/prompts/`
  放了一些 prompt 文本模板，供 baseline / SFT / RL 阶段使用。

---

## 2. 先建立一个核心认识

### 这个作业的“主入口”不是业务代码，而是 adapter

最关键的文件是：

- [tests/adapters.py](/home/cwoloc/CS336/assignment5-alignment/tests/adapters.py)

这个文件不是最终实现，而是一组“接口占位符”。
它定义了：

- 你要实现哪些函数
- 每个函数的参数和返回值是什么
- 测试会如何调用这些函数

你可以把它理解成：

- 测试层和你代码之间的桥
- 这门作业真正的 API 规范

如果你对整个项目只先读一个文件，那最推荐先读它。

---

## 3. 主作业最关键的文件

### `tests/test_sft.py`

文件：
- [tests/test_sft.py](/home/cwoloc/CS336/assignment5-alignment/tests/test_sft.py)

作用：
- 定义了 SFT 相关单元测试
- 说明你在主作业前半段要先实现什么

它主要测试这些函数：

- `run_tokenize_prompt_and_output`
- `run_compute_entropy`
- `run_get_response_log_probs`
- `run_masked_normalize`
- `run_sft_microbatch_train_step`

你可以把它理解成：
- “监督微调路径”的最小实现清单

### `tests/test_grpo.py`

文件：
- [tests/test_grpo.py](/home/cwoloc/CS336/assignment5-alignment/tests/test_grpo.py)

作用：
- 定义了 GRPO / policy gradient 相关单元测试

它主要测试这些函数：

- `run_compute_group_normalized_rewards`
- `run_compute_naive_policy_gradient_loss`
- `run_compute_grpo_clip_loss`
- `run_compute_policy_gradient_loss`
- `run_masked_mean`
- `run_grpo_microbatch_train_step`

你可以把它理解成：
- “强化学习路径”的最小实现清单

### `tests/conftest.py`

文件：
- [tests/conftest.py](/home/cwoloc/CS336/assignment5-alignment/tests/conftest.py)

作用：
- 定义测试用的 fixture
- 提供测试里反复使用的模型、tokenizer、张量、超参数

如果你看测试时不知道一个变量从哪来，大概率答案都在这里。

---

## 4. 你最应该优先理解的函数

下面按重要性整理主作业里的关键函数。

---

## 5. SFT 路线关键函数

这些函数都定义在 [tests/adapters.py](/home/cwoloc/CS336/assignment5-alignment/tests/adapters.py)。

### `run_tokenize_prompt_and_output`

功能：
- 把 `prompt_strs` 和 `output_strs` token 化
- 生成训练所需的 `input_ids`
- 生成语言模型训练标签 `labels`
- 生成 `response_mask`

为什么重要：
- 它决定了“哪些 token 算 prompt，哪些 token 算 response”
- 后面 loss、log prob、masking 都依赖它

关键输入：
- `prompt_strs: list[str]`
- `output_strs: list[str]`
- `tokenizer`

关键输出：
- `input_ids`
- `labels`
- `response_mask`

你可以把它理解成：
- 把“字符串样本”变成“训练张量样本”的入口

### `run_compute_entropy`

功能：
- 对 logits 最后一维计算熵

为什么重要：
- 熵常被用来观察模型预测分布是否过于尖锐或过于随机
- 在 RL 和训练分析里经常是辅助监控指标

关键输入：
- `logits: Tensor`

关键输出：
- 与 token 位置对应的 entropy tensor

### `run_get_response_log_probs`

功能：
- 让模型对 `input_ids` 前向计算
- 取出 `labels` 对应 token 的条件 log-prob
- 可选返回每个位置的 token entropy

为什么重要：
- 这是 SFT 和 RL 共用的核心底层函数
- 后续很多 loss 本质上都是在处理 token log-probs

关键输入：
- `model`
- `input_ids`
- `labels`
- `return_token_entropy`

关键输出：
- `log_probs`
- 可选 `token_entropy`

你可以把它理解成：
- “模型给这些目标 token 打了多少分”

### `run_masked_normalize`

功能：
- 只在 `mask == 1` 的位置上求和
- 再按给定常数归一化

为什么重要：
- 训练时通常只想统计 response 部分，不想把 prompt 或 padding 算进去
- 这是 SFT loss 聚合的基础工具函数

关键输入：
- `tensor`
- `mask`
- `dim`
- `normalize_constant`

关键输出：
- 归一化后的 masked sum

### `run_sft_microbatch_train_step`

功能：
- 对一个 microbatch 计算 SFT loss
- 结合 `response_mask`
- 进行梯度累计缩放
- 执行 `backward()`

为什么重要：
- 这是监督微调训练中最直接的“单步更新逻辑”

关键输入：
- `policy_log_probs`
- `response_mask`
- `gradient_accumulation_steps`
- `normalize_constant`

关键输出：
- `loss`
- metadata

你可以把它理解成：
- SFT 训练环节里最小的一步

---

## 6. RL / GRPO 路线关键函数

### `run_compute_group_normalized_rewards`

功能：
- 对 rollout responses 逐个调用 `reward_fn`
- 按 `group_size` 分组
- 计算组内归一化 rewards / advantages

为什么重要：
- 这是 GRPO 的核心思想入口
- 它把“每条回答的 reward”变成“相对同组其他回答的 advantage”

关键输入：
- `reward_fn`
- `rollout_responses`
- `repeated_ground_truths`
- `group_size`
- `advantage_eps`
- `normalize_by_std`

关键输出：
- `normalized_rewards`
- `raw_rewards`
- `metadata`

### `run_compute_naive_policy_gradient_loss`

功能：
- 用最基本的 policy gradient 公式计算逐 token loss

为什么重要：
- 它是最基础的 RL loss 版本
- 更复杂的版本可以看作它的扩展

关键输入：
- `raw_rewards_or_advantages`
- `policy_log_probs`

关键输出：
- per-token policy gradient loss

### `run_compute_grpo_clip_loss`

功能：
- 计算 GRPO-Clip 风格 loss
- 使用新旧策略 log-prob 的比值做 clipping

为什么重要：
- 这是主作业 RL 部分最核心的一个 loss
- 它和 PPO / clipped objective 的直觉很接近

关键输入：
- `advantages`
- `policy_log_probs`
- `old_log_probs`
- `cliprange`

关键输出：
- clipped per-token loss
- metadata

### `run_compute_policy_gradient_loss`

功能：
- 根据 `loss_type` 调度不同 loss 实现

支持的 `loss_type`：
- `"no_baseline"`
- `"reinforce_with_baseline"`
- `"grpo_clip"`

为什么重要：
- 这是 loss 选择层
- 训练循环里不必直接写死某一种 RL loss

### `run_masked_mean`

功能：
- 在 mask 指定位置上做 masked mean

为什么重要：
- RL loss 计算后，往往还要对 token 维或 batch 维做 masked 平均
- 它和 `run_masked_normalize` 是一组很常用的聚合工具

### `run_grpo_microbatch_train_step`

功能：
- 对一个 microbatch 计算 RL / GRPO loss
- 基于 `response_mask` 聚合
- 支持多种 `loss_type`
- 做反向传播

为什么重要：
- 它是 RL 训练最小执行单元

关键输入：
- `policy_log_probs`
- `response_mask`
- `gradient_accumulation_steps`
- `loss_type`
- 视情况还会用到 `raw_rewards`、`advantages`、`old_log_probs`、`cliprange`

关键输出：
- `loss`
- metadata

---

## 7. Optional 部分函数

这些也是在 `tests/adapters.py` 里，但属于 supplement，可先不优先处理。

### `get_packed_sft_dataset`

功能：
- 把 instruction tuning 数据打包成定长语言模型训练样本

### `run_iterate_batches`

功能：
- 给 `Dataset` 提供一个按 batch 迭代的接口

### `run_parse_mmlu_response`

功能：
- 把自由文本回答解析成 `A/B/C/D`

### `run_parse_gsm8k_response`

功能：
- 从输出里取最后一个数值答案

### `run_compute_per_instance_dpo_loss`

功能：
- 计算单个偏好样本上的 DPO loss

---

## 8. 关键类与工具

### `NumpySnapshot`

定义位置：
- [tests/conftest.py](/home/cwoloc/CS336/assignment5-alignment/tests/conftest.py)

功能：
- 把测试输出和 `tests/_snapshots/*.npz` 里的参考结果比较

你要知道的重点：
- 很多测试不是比“类型对不对”，而是直接比数值结果
- 所以函数实现只要公式略偏一点，snapshot test 就会失败

### `Snapshot`

定义位置：
- [tests/conftest.py](/home/cwoloc/CS336/assignment5-alignment/tests/conftest.py)

功能：
- 对非数组对象做 pickle-based snapshot 对比

在主作业里它不像 `NumpySnapshot` 那么核心，但原理类似。

---

## 9. 测试里最常见的关键变量

这些变量大多来自 [tests/conftest.py](/home/cwoloc/CS336/assignment5-alignment/tests/conftest.py)。

### `model_id`

值：
- `"/data/a5-alignment/models/Qwen2.5-Math-1.5B"`

含义：
- 主作业默认使用的基础模型路径

### `tokenizer`

来源：
- `AutoTokenizer.from_pretrained(model_id)`

含义：
- 与主模型配套的 tokenizer

### `model`

来源：
- `AutoModelForCausalLM.from_pretrained(model_id)`

含义：
- 测试使用的语言模型

### `prompt_strs`

含义：
- 一组简单示例 prompt 字符串

用途：
- 用于测试 tokenization 逻辑

### `output_strs`

含义：
- 一组简单示例输出字符串

用途：
- 用于测试 tokenization 和 labels / mask 构造

### `logits`

形状：
- `(batch_size, seq_length, vocab_size)`

用途：
- 测试 entropy 计算

### `input_ids`

形状：
- `(batch_size, seq_length)`

用途：
- 测试模型 log-prob 计算

### `labels`

来源：
- 对 `input_ids` 做右移构造

含义：
- 语言模型的 next-token prediction labels

### `policy_log_probs`

形状：
- `(batch_size, seq_length)`

用途：
- 测试 SFT 和 GRPO 的 token 级损失

### `response_mask`

注意：
- 这个变量在 `test_sft.py` / `test_grpo.py` 的测试参数里会出现
- 它表示哪些 token 属于 response，哪些不属于

作用：
- 训练时只在 response token 上计算有效损失

### `raw_rewards`

含义：
- rollout 的原始奖励

### `advantages`

含义：
- 常见是 reward 做 baseline / 组归一化后得到的优势值

### `old_log_probs`

含义：
- 旧策略对同一批 token 的 log-prob

用途：
- GRPO-Clip 需要新旧策略比值

### `cliprange`

含义：
- clipping 范围超参数

---

## 10. Prompt 文件速记

这些文件在：
- [cs336_alignment/prompts](/home/cwoloc/CS336/assignment5-alignment/cs336_alignment/prompts)

### `r1_zero.prompt`

作用：
- 主作业 zero-shot / reasoning baseline 常用 prompt 模板

### `question_only.prompt`

作用：
- 只给问题本体的简化 prompt 模板

### `zero_shot_system_prompt.prompt`

作用：
- supplement 的 general assistant system prompt

### `alpaca_sft.prompt`

作用：
- instruction tuning 风格的 SFT prompt 模板

---

## 11. 现有源码中少数真正有逻辑的业务文件

### `cs336_alignment/drgrpo_grader.py`

文件：
- [cs336_alignment/drgrpo_grader.py](/home/cwoloc/CS336/assignment5-alignment/cs336_alignment/drgrpo_grader.py)

作用：
- 提供数学答案评分 / 归一化相关逻辑
- 更像一个可参考工具，而不是你主线必须先读完的文件

如果你后面在 reward function 或答案判定上遇到疑惑，再回来看它会更有价值。

---

## 12. 这份代码库当前“没有”的东西

理解这一点很重要，不然你会一直找“现成实现”。

目前仓库里基本没有：

- 完整的 SFT trainer
- 完整的 GRPO trainer
- 你要实现的核心函数的参考源码
- 可直接运行的完整主线训练框架

也就是说：

- 这是“框架 + 测试 + 文档”型作业
- 不是“读懂现成系统再改 bug”型作业

---

## 13. 初学者最推荐的阅读顺序

### 第一遍：只看地图

1. 看 [tests/test_sft.py](/home/cwoloc/CS336/assignment5-alignment/tests/test_sft.py)
2. 看 [tests/test_grpo.py](/home/cwoloc/CS336/assignment5-alignment/tests/test_grpo.py)
3. 看 [tests/adapters.py](/home/cwoloc/CS336/assignment5-alignment/tests/adapters.py)
4. 看 [tests/conftest.py](/home/cwoloc/CS336/assignment5-alignment/tests/conftest.py)

目标：
- 知道有哪些函数要实现
- 知道输入输出长什么样

### 第二遍：只看一条主线

先只跟 SFT：

1. `run_tokenize_prompt_and_output`
2. `run_compute_entropy`
3. `run_get_response_log_probs`
4. `run_masked_normalize`
5. `run_sft_microbatch_train_step`

目标：
- 先跑通监督学习路径

### 第三遍：再看 RL

1. `run_compute_group_normalized_rewards`
2. `run_compute_naive_policy_gradient_loss`
3. `run_compute_grpo_clip_loss`
4. `run_compute_policy_gradient_loss`
5. `run_masked_mean`
6. `run_grpo_microbatch_train_step`

目标：
- 再拼出 RL 路线

---

## 14. 一句话版本记忆卡

如果你只想先记住最核心的映射关系，可以记这个：

- `tests/test_sft.py`：告诉你怎么做 SFT
- `tests/test_grpo.py`：告诉你怎么做 RL / GRPO
- `tests/adapters.py`：告诉你每个函数该长什么样
- `tests/conftest.py`：告诉你测试输入数据长什么样
- `tests/_snapshots/`：告诉你数值结果应该长什么样
- `cs336_alignment/`：留给你自己实现

---

## 15. 你下一步最推荐做什么

最推荐的下一步不是继续泛读整个仓库，而是做下面这件事：

1. 打开 [tests/test_sft.py](/home/cwoloc/CS336/assignment5-alignment/tests/test_sft.py)
2. 盯住第一个函数 `run_tokenize_prompt_and_output`
3. 对照 [tests/adapters.py](/home/cwoloc/CS336/assignment5-alignment/tests/adapters.py) 的 docstring 理解输入输出
4. 再决定你要把实现写进 `cs336_alignment/` 的哪个模块

这是从“知道全局”过渡到“开始动手”的最顺滑路径。
