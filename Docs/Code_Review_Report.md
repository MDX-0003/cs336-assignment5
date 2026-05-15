# adapters.py



```
uv run pytest tests/test_sft.py::test_get_response_log_probs -q
uv run pytest tests/test_sft.py -k masked_normalize -q

```



## run_tokenize_prompt_and_output()

### 理念讨论

**SFT训练不希望模型因为prompt部分被扣分。**这句话同时隐含两个意思：

1.模型能够看到的不仅仅是O0、O1、O2这些“之前推理结果”，还包含用户的prompt。并且SFT不希望这些位置有loss

2.response_masks 用来区分那些“属于回答”的token，只有这些才会参与loss计算



**如果 labels 中某个位置是 prompt token，比如 P1，它算不算“正确 token”？**

答案：从 next-token prediction 的角度，它是正确 token。因为看到 P0 后，原始序列的下一个 token 的确是 P1。

但从 SFT 的目标来看，它不是我们要优化的 response token，所以它会被 response_mask=False 排除。



### 1

```
wrong:
token_prompts = tokenizer.encode(prompt for prompt in prompt_strs)


right:
token_prompts  = (tokenizer.encode(prompt ) for prompt in prompt_strs)
```

encode需要的是str，而非for产生的generater

另外不要使用()，list要用[]



另外还可以用batch 写法：

```
token_prompts = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
token_outputs = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
```

tokenizer(....)是在调用已经创建的对象，["input_ids"]是将其中某一项提取出来。

这种做法比for效率更高，但是要记住不要添加特殊token



### 2

对于两个list分别存储一批prompt和output，直接相加和逐seq拼接是不同的

```
full_tokens = token_prompts + token_outputs
假设token_prompts  = [[P0, P1], [P0, P1, P2]]
token_outputs = [[O0], [O0, O1]]
直接相加[[P0, P1], [P0, P1, P2], [O0], [O0, O1]]
```

两层循环也不对，正确的做法是zip，能实现两个list的逐元素拼接

zip里顺序很重要，不能反写

```
full_tokens = [
    prompt_tokens + output_tokens
    for prompt_tokens, output_tokens in zip(token_prompts, token_outputs)
]
```



### 3

字符串换行自动拼接，不加逗号

f前缀能够把变量直接输出到str

```
if len(prompt_strs) != len(output_strs):
        raise ValueError(
            f"xxx"
            f"yyy"
        )
```



### 4 

list 即[xxx,xxx]可以用乘号表示重复

下例里如果pad_token_id直接乘，只会让int乘以倍数改变数值

```
token_need_pad = [pad_token_id] * (max_seq_len - len(full_token))
full_token = full_token + token_need_pad
```



## run_compute_entropy

### 1

softmax = 某一logit指数/所有logit指数和，softmax以后概率已经从[-inf , + inf]缩放为求和 = 1、范围[0,1]。

之所以还要叠一次log，是为了保证过小的softmax结果能够被缩放到正常数值。

log_softmax是优化后接口，比分开做更稳定。两个接口都必须指定dim = -1

这里求的是信息熵，公式为:
$$
(H = -\sum p \log p)
$$
所以算两份log相乘

## run_get_response_log_probs 

考虑到response_masks ，run_get_response_log_probs 需要计算：

```
target_log_probs[t] = log P(labels[t] | input_ids[:t+1])
```

上述公式理解：输入为前t个token（input_ids[:t+1]），按照训练数据，正确的下一tokens是labels[t] ，那么模型说对的概率（取对数）是多少？

```
full_tokens = [P0, P1, P2, O0, O1]
input_ids      = [P0, P1, P2, O0]
labels         = [P1, P2, O0, O1]
response_mask  = [ 0,  0,  1,  1]
在上述条件下：
log_probs = [
    log P(P1 | P0),
    log P(P2 | P0 P1),
    log P(O0 | P0 P1 P2),
    log P(O1 | P0 P1 P2 O0),
]
参与loss计算的，只有response_mask=True对应的那些probs
于是SFT只用后两个
log P(O0 | P0 P1 P2)
log P(O1 | P0 P1 P2 O0)
```

### 1

给dict赋值未指定的ket value对，是可行的

```
    result = {"log_probs":target_log_probs}
    if return_token_entropy :
        result["token_entropy"] = run_compute_entropy(logits)
```

### 2

这段的意思是：log_probs  [B,L,V] ，每个token都有一个对数概率，现在要取里面正确token的概率，labels的每个元素正好就是token id，可以作为dim = - 1的index用来取值。

取值前要对labels升维，取值后结果要降维，要和entropy形状一致（每个seq有唯一的一个熵）

```
target_log_probs=torch.gather(log_probs,dim = -1,index = labels.unsqueeze(-1))
    target_log_probs = target_log_probs.squeeze(-1)
    # labels[B,L] -> [B,L,1] ,log_probs [B,L,V] ,target_log_probs [B,L,1]->[B,L]
```





## run_masked_normalize

现在有

```
policy_log_probs: 每个 token 位置的 log-prob
response_mask:    哪些 token 位置属于 response
```

在当前函数内，我们要筛出属于response的log-prob，对他们进行归一化

数值例子

```
tensor = [1.0, 2.0, 3.0, 4.0]
mask   = [True, False, True, False]
normalize_constant = 2
tensor * mask = [1.0, 0.0, 3.0, 0.0] 
求和归一化即可
```

为了保证mask 和 tensor能够正确相乘，要让前者能够匹配后者的数据类型

有

```
masked_tensor = tensor * mask.to(dtype=tensor.dtype)
return masked_tensor.sum(dim=dim) / normalize_constant
```



## run_sft_microbatch_train_step

**为什么有micro Batch？**

因为一个batch可能太大，显存装不下，所以一次运行会拆成大约8个microBatch，并且每个microBatch只累积梯度（不清零）不反向传播，最终把8个 loss求和再除以8，得到整个Batch的loss

**主要的起步Tensor**

- policy_log_probs[b, t]   

  第 b 条样本，第 t 个位置，模型给 labels[b, t] 这个正确 token 的 log 概率

  概率是0-1的，log_prob以后范围是[-inf,0]，概率越接近1，log_prob就约接近0

  训练的目标是最大化log_prob，但是optimizer默认会最小化loss

  所以loss 就 等于 -log_prob，也叫做对数似然 NLL (negative log likelyhood)

- response_mask [b,t]

  哪些 token 属于 response，要参与训练 

**如果prob变大，那么nll变大还是变小？**

从0.01 到 0.5，nll变小了，因此最小化nll（loss）就是目标

```
-log(0.01) ≈ 4.605
-log(0.5)  ≈ 0.693
```

**计算loss的过程：**

1-3是前文 `run_masked_normalied`的实现逻辑

4-6是当前microBatch loss的构造，除以8是为了最后加起来得到batch的 loss

```
1. 对每条样本，只保留 response_mask=True 的 log_probs
2. 每条样本内部求和
3. 除以 normalize_constant

4. 对 batch 求平均
5. 加负号，变成要最小化的 SFT loss
6. 除以 gradient_accumulation_steps

7. 调用 backward()
8. 返回 loss 和 metadata
```

**关于梯度**

microBatch里做什么：

- 算loss，并且loss.backward()，把当前loss的梯度，累积到每个参数

microBatch不做什么：

- 不调用optimizer.step() , 要等所有microBatch做完了，再更新参数
- 不调用optimizer.zero_grad() ,要等所有microBatch做完了，再清空梯度



## run_masked_mean

和前文的run_mask_normalize对比一下，相同点：

二者都只对属于response的token计算归一化（tensor要先乘以mask，再sum）

不同点：normalize有专门的参数作为sum以后的分母

mean需要以response token的数量作为分母（直接对01 的mask求和就能得到· ）

这里的难点在于要考虑dim，dim不为0时，要对dim求mean

```
    mask_tensor = tensor*mask.to(dtype = tensor.dtype)
    return mask_tensor.sum(dim=dim)/mask.sum(dim=dim).to(dtype = tensor.dtype)
```

**为什么要做两个mask_mean？**

normalize的版本（手动指定归一化程度）用于SFT，而mean版本用于GRPO

GRPO 的 microbatch train step 会先得到**每个 token 的 policy gradient loss**，然后只在 response token 上求平均。



**如果tensor维度为[2,3]dim=1，那么mean以后，维度是多少？**答案是[2,]

```
tensor =
[
  [1, 2, 3],
  [4, 5, 6],
]

mask =
[
  [1, 0, 1],
  [0, 1, 1],
]
tensor * mask
第一行 mean = (1 + 3) / 2 = 2
第二行 mean = (5 + 6) / 2 = 5.5
输出[2, 5.5]
```

**和教案示例的区别：**

前面的代码能够通过git教案的test，但是我们尽量做的更好：

假如mask全0（大概率不会发生），则求mean时，分母的mask.sum()就可能为0，我们需要限制一下

另外，to的格式转换最好先做，以免sum不了

```
mask.sum(dim=dim).to(dtype = tensor.dtype)
mask.sum(dim=dim).to(dtype = tensor.dtype).clamp_min(1e-8)
mask.to(dtype = tensor.dtype).sum(dim=dim).clamp_min(1e-8)
```

最后测试时，发现加的这一步clamp是多此一举。如果加了，除0时，答案就是0.如果不加，就因为非法除0得到NaN。

测试代码期待一个NaN，因此去掉clamp更好

## run_compute_group_normalized_rewards

听名字就知道，组内归一化奖励。

**什么叫组内？**

针对同一个问题，模型要回答4次，得到4个reward，但是不直接返回means。而是计算组内advantage，也就是对于4次回答，每次回答都有一个advantage ，表示**“当前回答比组内平均回答好多少？”**

这个优势常写作：r_i即每个回答的reward，
$$
A^ i=r_i−μ_r
$$
公式有两个版本，根据输入参数不同，决定要不要标准化（除以标准差）

```
advantage = reward - group_mean
advantage = (reward - group_mean) / (group_std + eps)
group_std = grouped_rewards.std(dim = -1,keepdim = True,unbiased=False)
```

**标准差是不是无偏（unbiased是不是False）（分母是不是N）**

假如我们把策略视作一个巨大分布，4次问答只是对这个巨大分布的一个估计，那么抽样的标准差就会用分母 = -1 、参数unbiased = True来写

GRPO里代码均使用unbiase = False，意味着我们始终将4个问答视作一个完整分布。

另外也有实际的原因，即如果问答次数为1，除以N-1会除以0，当然代码里会使用eps来解决。

**参数里的单词含义**

一个单词要问好几次，就叫 rollout， rollout_response的长度即 **每个问题问几次** * **问题的个数** 。rollout_batch_size = **n_prompts_per_rollout_batch * group_size**

repeated_ground_truths 长度同上，这意味着里面会有多组内容一样的元素，因为同一个问题，正确答案不会变，问4次，就重复给4次正确答案

```
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
```

**为什么是同一个问题作为一组？**

因为题目难度不同，模型在其上的策略分布就不同。为了独立计算advantage，GRPO关注“同一道题内部，谁比同组其它回答更好？”



### 1

如果想for 遍历一组list，取其元素作为另一个function的参数，可以把for写在括号[]里面，而不是把function写在for里面

```
rewards=[]
    for response,ground_truth in zip(rollout_responses,repeated_ground_truths):
        reward = reward_fn(response,ground_truth)
        rewards.append(reward)
```

两段代码等价，不用写额外变量

```
  reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
```



## run_compute_naive_policy_gradient_loss

```
 """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
```

首先需要明确reward只有一个维度，所有问题*所有回答，全部摊平到一维

policy_log_probs 是模型最终输出的每一个token，选择到的token的对数概率

这个函数的功能是把两者相乘，reward[B,1]，log_prob[B,T]，意味着**给每个最终选定的prob一个来自reward的权重，如果奖励高，那么这个被选中的token未来就更应该被选中。**

**参数来源 为什么叫reward or advantage？**

取决于未来loss的构造方法，依方法不同，使用不同的权重来源

本函数不关心权重来源，他只会把权重乘算到log_prob上

公式始终是：

```
loss = -A * log_prob
```



- loss_type="no_baseline" 

  raw_reward来自reward_fn的直接输出，也可以是上一个函数run_compute_group_normalized_rewards输出的子项

  此时权重表示**“这个回答在reward_fn看来好多少”**

- loss_type = "reinforce_with_baseline"

  权重来源于组内优势估计，可以是上一个函数run_compute_group_normalized_rewards输出的子项

  此时权重表示**“这个回答比同组好多少”**

- loss_type = "grpo_clip"

  权重依然来自advantage，考虑到grpo公式里的clip，还需要传旧的old_log_probs 和 cliprange



## run_compute_grpo_clip_loss

这个函数和前一个的差距是，本函数严格参考了GRPO公式，添加了clip机制，前函数只是把reward_or_advantage作为权重乘在log_prob上（[B,1] * [B,t]）

完整的GRPO loss可以写成：

```
L_i,t = - min(
    r_i,t * A_i,
    clip(r_i,t, 1 - ε, 1 + ε) * A_i
)
r_i,t = π_θ(o_t | q, o_<t) / π_θ_old(o_t | q, o_<t)
r_i,t是重要性采样的权重
q，prompt/提问
o，llm的回答/response
o_t，回答里第t个token
o_<t，0到t-1个token
A_i，第i条样本的advantage
	注意A_i = 样本得分-组内得分
ε，clip范围
```



**和naive版本相比，为什么loss要加一个clip？**

clip的本质是对重要性采样(r_i,t)的一次包装，因此关键在于为什么要给loss一个新策略/旧策略的比值？

目标是控制新旧策略的更新比例。

**策略到底是什么？一个数？一组数？**

算GRPO loss时，策略就是log_prob，被选中token的logits概率

**新旧策略的重要性权重是prob相除吗？**

是原始prob相除，因此可以让新旧log_prob相减，再放进exp

```
ratio = torch.exp(policy_log_probs - old_log_probs)
```

**为什么公式有个min**

min表示如果ratio的权重很大，那么模型会更新得很激进

此时就用clip后的ratio来替换

```
loss = - min(ratio * A, clipped_ratio * A)
```

**naive loss和 clip loss是互相替代的关系吗**

```
naive loss = -A * log_prob
clip loss = -min(ratio * A, clipped_ratio * A)
```

前者表示当前token的log_prob要不要被奖励

后者表示当前策略（log_prob）相对旧的策略，要更新多少



**除了loss，这个函数还返回一个metadata是什么？**

clip_fraction，一句回复里，T个token，有多少个在更新时是clip过的？

显然这个问题在问：下面loss计算时，当前token选的是左边还是右边？

因此需要判断一下ratio和clip_ratio哪个更小

```
clip loss = -min(ratio * A, clipped_ratio * A)
```



### 1

两个同尺寸tensor，求两边中最小值来构造新tensor，不能像cpu端list那样写min，要写：

```
loss = -torch.minimum(raw_objective,clip_objective)
```





## compute_policy_gradient_loss

这个函数只做聚合，根据不同的loss type，选择loss

总共三种：

- no_baseline : run_compute_naive_policy_gradient_loss

  

- reinforce_with_baseline:run_compute_naive_policy_gradient_loss

- grpo_clip: run_compute_grpo_clip_loss

回顾一下，naive的公式始终是：，前两个type的区别在于A选什么，是raw reward 还是 raw reward - 组内平均reward (组内优势)

```
loss = -A * log_prob(π,当前策略)
loss = -min(ratio * A, clipped_ratio * A)
```

一对比就可以明确发现，**GRPO的loss，A只乘比例，是不乘以log_prob的**

## run_grpo_microbatch_train_step

直到per-token-loss为止，都是比较熟悉的。后续的步骤就是把response部分提出来做mean，得到真正的loss

```
policy_log_probs
raw_rewards / advantages / old_log_probs
        |
        v
run_compute_policy_gradient_loss
        |
        v
per_token_loss: shape (B, T)
        |
        v
response_mask 只保留 response token
        |
        v
masked_mean 得到 scalar loss
        |
        v
loss / gradient_accumulation_steps
        |
        v
backward()

```

