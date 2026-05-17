import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json

from vllm import LLM, SamplingParams


# ====================== Eval Row / Result Schema ======================

#every question with its answer should be store in class like this
@dataclass
class EvalRow:
    idx: int
    problem_id: Optional[str]
    prompt: str
    ground_truth: Any
    response: str
    category: str  # "F1A1", "F1A0", "F0A0"
    reward: float
    format_reward: float
    answer_reward: float


# ====================== JSONL IO ======================

def load_jsonl(path:str|Path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path: str, rows: List[EvalRow]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


# ====================== MATH Field Helpers ======================

# MATH jsonl usually has "problem" / "question"or similar
def get_question(example: Dict[str, Any]) -> str:
    for key in ["problem", "question", "prompt"]:
        if key in example and isinstance(example[key], str):
            return example[key]
    raise KeyError(f"Cannot find question field in example, actual keys={list(example.keys())}")


# MATH jsonl usually has "answer" or similar
# Answer can be str/int/float sowe use any
def get_ground_truth(example: Dict[str, Any]) -> Any:
    for key in ["answer", "ground_truth", "target"]:
        if key in example:
            return example[key]
    raise KeyError(f"Cannot find answer field in example, actual keys={list(example.keys())}")


def get_category(format_reward:float,answer_reward:float):
    if format_reward == 1.0 and answer_reward == 1.0:
        return "F1A1"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "F1A0"
    return "F0A0"


# ====================== vLLM Evaluation ======================

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, Any], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[Any],
    sample_params: SamplingParams,
    request_batch_size: int = 64,
) -> List[EvalRow]:
    assert len(prompts) == len(ground_truths)
    eval_rows = [] #List[EvalRow]
    n = len(prompts)
    for start in range(0,n,request_batch_size):
        end = min(start + request_batch_size,n)
        cur_prompt = prompts[start:end]
        cur_gt = ground_truths[start:end]

        outputs = vllm_model.generate(cur_prompt,sample_params)
        #outputs List[RequestOutput]
        assert len(cur_prompt) == len(outputs)

        for i,request_output in enumerate(outputs):
            assert request_output.prompt == cur_prompt[i]
            prompt = request_output.prompt
            output = request_output.outputs[0].text
            gt = cur_gt[i]
            scores = reward_fn(output,gt) # Dict[str,float] reward/format_reward/answer_reward
            reward = float(scores.get("reward",0.0))
            format_reward = float(scores.get("format_reward",0.0))
            answer_reward = float(scores.get("answer_reward",0.0))
            cur_metadata = EvalRow(
                idx = start+i,
                problem_id = None,
                prompt = prompt,
                ground_truth = gt,
                response = output,
                category = get_category(format_reward,answer_reward),
                reward=reward,
                format_reward=format_reward,
                answer_reward=answer_reward
            )
            eval_rows.append(cur_metadata)
    return eval_rows


# ====================== Eval Summaries / Samples ======================

def summarize(rows:List[EvalRow])->Dict[str,Any]:
    summarized_rows = []
    n = len(rows)
    c = {"F1A1": 0, "F1A0": 0, "F0A0": 0}
    for row in rows:
        c[row.category] += 1
    format_rate = sum(row.format_reward for row in rows)/n
    answer_accuracy =  sum(row.answer_reward for row in rows)/n
    avg_reward = sum(row.answer_reward for row in rows)/n
    return  {
        "n": n,
        "counts": c,
        "format_rate": format_rate,
        "answer_accuracy": answer_accuracy,
        "avg_reward": avg_reward,
    }


def sample_examples(
        rows: List[EvalRow],
        category: str, k: int = 10
    ) -> List[EvalRow]:
    picked = [r for r in rows if r.category == category]
    return picked[:k]
