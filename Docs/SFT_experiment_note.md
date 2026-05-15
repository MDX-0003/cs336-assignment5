从jsonl里获取数据，每一行str作为list的每个元素：

```
def load_jsonl(path: str | Path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)
examples = list(load_jsonl(data_path))[:8]
```

- yield保证jsonl的内容不会一次性进内存，内存里只占据当前读的这一行的空间。

  注意yield和return不能一起用，只能二选一

- "path: str | Path"表示参数必须是二者其一



读取prompt模板+应用到真实prompt

```
prompt_template = Path(prompt_path).read_text()
prompts = [
        prompt_template.format(question=example["problem"])
        for example in examples
    ]

```

- Path对象可以直接调用read_text()，返回一个str（提示词模板）

- str.format就是把str里，用{xxxx}包裹起来的片段，替换为其他变量。这就实现了prompt的应用

  ```
  prompt_template = "请回答这个问题：{question}"
  example["problem"] = "1+1等于几？"
  
  prompt_template.format(question=example["problem"])
  "请回答这个问题：1+1等于几？"
  ```

  

  

vllm.LLM.generate的返回值到底是什么？

RequestOutput类型，`https://docs.vllm.ai/en/latest/api/vllm/#vllm.RequestOutput`

```
RequestOutput成员

request_id:str
prompt:str
prompt_token_ids:list[int]
prompt_logprobs : PromptLogprobs (专门的class)
outputs: list[CompletionOutput]
还有一个bool

```



```
CompletionOutput成员

index：int//这个output是整个llm RequestOutput的第几个
text:str
token_ids:Sequence[int]
```

关键是最后一个成员outputs，包含多个CompletionOutput，一般只有一个，所以都去拿outputs[0].text

```

    rewards = []
    for example, output in zip(examples, outputs):
        response = output.outputs[0].text
        ground_truth = example["answer"]

        reward = r1_zero_reward_fn(response, ground_truth)
        rewards.append(reward)

```





- r1_zero_reward_fn 要求格式为：

  ```
  "</think> <answer>" in response and "</answer>" in response
  ```

  

- prompt格式里，结尾始终为：

  ```
  Assistant: <think>
  ```

  

- 当vllm的sample_params设置了以后

  ```
  sample_params = SamplingParams
  	(
  		...
  		stop=["</answer>"],
  		include_stop_str_in_output = True,
  	)
  ```

  每次推理最后一定会以</answer>结尾

- 上述三者是相匹配的，reward_fn最后必须看见</answer>，我们输入的prompt以<think>开头，那么就应该期待llm主动给出</think>

