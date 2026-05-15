# CS336 Assignment 5 初学者执行清单

这份清单的目标不是替代原始 handout，而是把主作业拆成一个更容易执行的路线图。

适用范围：
- 以主文档 `Docs/cs336_spring2025_assignment5_alignment.md` 为主
- 默认你先完成必做部分
- 默认你是第一次做这类“语言模型微调 + 强化学习”课程作业

---

## 1. 先理解这门作业在做什么

你要做的是把一个基础数学模型 `Qwen 2.5 Math 1.5B Base` 按下面顺序一步步变强：

1. 先测它什么都不训练时的表现
2. 再用监督学习让它模仿更强模型的推理过程
3. 再用 Expert Iteration 让它“自己做题、自己筛选正确轨迹、继续学习”
4. 最后用 GRPO 这种 RL 方法继续提升解题能力

你最终交的主要是两样东西：
- `writeup.pdf`
- `code.zip`

---

## 2. 你现在最应该先知道的几个词

### Zero-shot baseline 是什么

意思是：

- 不做任何额外训练
- 直接拿现成基础模型
- 给它 prompt，让它做题
- 统计它本来就有多强

这个 baseline 的作用是：

- 作为后续 SFT、EI、GRPO 的对照组
- 告诉你“训练前”的起点是多少
- 帮你确认评测代码和答案解析代码是通的

如果你后面训练完模型，却没有比 zero-shot baseline 更好，那通常说明：

- 训练没有真正起作用
- reward / parsing / evaluation 可能写错了
- 超参数可能不合适

### SFT 是什么

SFT 是监督微调。简单说，就是拿“输入 -> 正确推理过程和答案”的样本来教模型模仿。

### Expert Iteration 是什么

模型先自己生成多条解题轨迹，然后系统用 reward 检查哪些是对的，把对的留下来继续训练。

### GRPO 是什么

这是一种强化学习训练方法。你不再只让模型模仿标准答案，而是根据模型生成结果的“好坏”来更新模型。

---

## 3. 初学者最推荐的完成顺序

严格建议按这个顺序走，不要一开始就在所有部分来回跳。

1. 环境和仓库结构跑通
2. 搞懂 zero-shot baseline 和评测
3. 完成 SFT 相关小函数和测试
4. 跑通完整 SFT
5. 完成 Expert Iteration
6. 完成 GRPO 的核心函数和训练循环
7. 跑实验、整理结果、写 writeup
8. 有余力再做 supplement

---

## 4. 第一阶段：先把环境和代码结构看懂

### 你先要看哪些文件

- `README.md`
- `Docs/cs336_spring2025_assignment5_alignment.md`
- `tests/test_sft.py`
- `tests/test_grpo.py`
- `tests/adapters.py`
- `cs336_alignment/`

### 这一阶段你的目标

- 知道代码该写在哪
- 知道测试怎么连接到你的代码
- 知道这次真正必过的是哪些测试

### 你要确认的事情

- 你能进入项目环境
- 你能运行 `uv run pytest`
- 你知道 `tests/adapters.py` 是测试入口桥梁
- 你知道主作业重点是 `tests/test_sft.py` 和 `tests/test_grpo.py`

---

## 5. 第二阶段：先把 zero-shot baseline 跑通

这一阶段的重点不是训练，而是“先把测量做好”。

### 你要完成什么

- 能加载基础模型
- 能给模型喂 prompt
- 能得到模型输出
- 能从输出里提取最终答案
- 能把预测答案和标准答案比较
- 能算出 baseline 表现

### 你在学什么

- prompt 长什么样
- MATH 数据怎么组织
- 模型输出是什么格式
- 评测为什么经常需要“解析答案”

### 你做完后应该能回答

- 模型完全没训练时，准确率 / reward 大概是多少
- 你的 evaluation pipeline 是否稳定
- 输出里哪些内容是推理，哪些内容是最终答案

### 自查标准

- 你知道 baseline 只是起点，不是最终目标
- 你可以解释为什么要先测 baseline
- 你可以复现一次 baseline 结果

---

## 6. 第三阶段：先做 SFT 小函数，再做完整训练

这一阶段建议你采用“测试驱动”的方式。

### 推荐做法

先实现一个小函数，然后立刻跑它对应的测试，不要写一大堆再统一调。

### 你通常会依次碰到的内容

- prompt 和 output 的 tokenization
- entropy 计算
- response log probabilities
- masked normalize
- SFT microbatch train step
- 完整 SFT train loop

### 这一阶段你的真正目标

- 不是背公式
- 而是理解“监督微调一次更新到底需要哪些张量”

### 做完 SFT 后你应该得到什么

- 一套能训练的 supervised pipeline
- 一个比 zero-shot 更强的模型
- 可以写进报告的验证集结果

---

## 7. 第四阶段：做 Expert Iteration

这一阶段开始进入“模型自己生成，再筛选”的范式。

### 你要理解的核心

- 模型先采样多条 reasoning trajectories
- 系统根据 reward 判断答案是否正确
- 只保留好的轨迹
- 再继续用这些好轨迹做训练

### 你应该重点关注

- reward function 是否正确
- 采样出来的答案怎么验证
- 保留下来的样本格式是什么
- 新数据如何重新喂回训练流程

### 初学者常见误区

- 只关注生成文本长不长，不关注答案是否可验证
- 没有先确保 reward function 可靠
- 采样和训练接口写得彼此不兼容

---

## 8. 第五阶段：做 GRPO

这是主作业里最难的一块，但你可以把它拆开理解。

### 你要实现的核心逻辑

- 计算 group-normalized rewards / advantages
- 实现 policy gradient 相关 loss
- 实现 GRPO-Clip loss
- 写 GRPO microbatch train step
- 写完整 GRPO train loop

### 这一阶段你真正要会的东西

- RL 训练时优化目标和 SFT 不一样
- 这里的“loss”很多时候只是为了反向传播方便，不是传统监督学习里的可解释 loss
- 训练时更应该重点看 reward、accuracy、entropy、稳定性曲线

### 你做实验时要重点记录

- 不同超参数下 reward 曲线
- 是否出现发散
- on-policy 和 off-policy 的区别
- 不同 normalization / clipping 方法的表现差异

---

## 9. 你每个阶段都应该怎么工作

推荐固定 workflow：

1. 先读 handout 对应小节
2. 找到对应测试
3. 找到 `tests/adapters.py` 里对应入口
4. 在 `cs336_alignment/` 里实现代码
5. 先跑局部测试
6. 通过后再进入下一小节

这个节奏对初学者很重要，因为它能避免你在大系统里迷路。

---

## 10. 写 writeup 时要记录什么

不要等全部做完才回忆，边做边记。

### 建议持续记录

- baseline 结果
- SFT 后的结果
- EI 后的结果
- GRPO 后的结果
- 每次关键实验的超参数
- 训练是否稳定
- 你观察到的现象
- 你对结果的解释

### 你可以随手建一个实验记录模板

- 实验名
- 模型
- 数据
- 关键超参数
- 验证集结果
- 是否异常
- 一句话结论

---

## 11. 初学者版本的实际执行清单

下面这部分你可以直接当待办事项来打勾。

### A. 启动阶段

- [ ] 阅读 `README.md`
- [ ] 阅读主文档前 3 个大部分，知道整体任务结构
- [ ] 打开 `tests/test_sft.py`
- [ ] 打开 `tests/test_grpo.py`
- [ ] 打开 `tests/adapters.py`
- [ ] 确认你知道代码要写在 `cs336_alignment/`

### B. Baseline 阶段

- [ ] 看懂 zero-shot prompt 长什么样
- [ ] 跑通一次模型生成
- [ ] 能读取一条 MATH 样本
- [ ] 能从模型输出中提取最终答案
- [ ] 能比较预测和标准答案
- [ ] 记录 zero-shot baseline 结果

### C. SFT 阶段

- [ ] 实现 tokenization 相关函数
- [ ] 实现 entropy 相关函数
- [ ] 实现 response log prob 相关函数
- [ ] 实现 normalize / masked 相关函数
- [ ] 实现 SFT microbatch step
- [ ] 跑对应单元测试直到通过
- [ ] 实现完整 SFT 训练循环
- [ ] 跑一次 SFT 实验并记录结果

### D. Expert Iteration 阶段

- [ ] 看懂生成轨迹 -> 验证 -> 筛选 -> 再训练的流程
- [ ] 确认 reward function 正确
- [ ] 跑通一轮小规模 EI
- [ ] 记录 EI 后效果是否优于 SFT

### E. GRPO 阶段

- [ ] 实现 group-normalized rewards
- [ ] 实现 naive policy gradient loss
- [ ] 实现 GRPO-Clip loss
- [ ] 实现 policy gradient loss wrapper
- [ ] 实现 masked mean
- [ ] 实现 GRPO microbatch step
- [ ] 跑对应测试直到通过
- [ ] 跑完整 GRPO 训练
- [ ] 记录 reward 曲线和验证结果

### F. 收尾阶段

- [ ] 汇总所有结果
- [ ] 回答书面问题
- [ ] 整理 `writeup.pdf`
- [ ] 整理 `code.zip`

---

## 12. 如果你时间不够，优先级怎么排

如果时间紧，优先顺序建议如下：

1. 先保证主作业必做测试能通过
2. 先保证 baseline、SFT、GRPO 主线能讲清楚
3. 再优化实验结果
4. 最后才做 supplement

不要把大量时间花在：

- 过早追求最好看的实验曲线
- 还没跑通测试就开始大规模训练
- 同时推进主作业和 supplement

---

## 13. 你完成主作业时，应该已经真正学会了什么

如果你顺利做完主文档，你应该已经掌握：

- 怎么评测一个语言模型在具体任务上的表现
- 怎么做基础的 supervised fine-tuning
- 怎么把“答案是否正确”写成 reward
- 怎么把语言模型训练扩展到 RL
- 为什么 baseline、SFT、EI、GRPO 是逐步递进的

这门作业的关键不是让你“背概念”，而是让你亲手把一条从 baseline 到 RL 的完整训练链路搭起来。

---

## 14. 你下一步最推荐做什么

如果你刚开始，最推荐的下一步是：

1. 先读主文档里 zero-shot baseline 和 SFT 的部分
2. 打开 `tests/test_sft.py`
3. 找到第一个要实现的 adapter
4. 先通过第一个最小测试

不要着急上 GRPO。先把 baseline 和 SFT 跑通，你会轻松很多。
