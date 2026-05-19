import os
import sys

# vllm 0.10.0 uses msgpack by default; collective_rpc with callable requires pickle fallback
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

from vllm import LLM,SamplingParams
import argparse
import gc
import logging
import random
import torch
from torch.utils.data import Dataset, DataLoader
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from cs336_alignment.sft_utils import (
    evaluate_vllm,
    get_ground_truth,
    get_question,
)
from tests.adapters import (
    run_tokenize_prompt_and_output as tokenize_prompt_and_output,
    run_get_response_log_probs as get_response_log_probs,
    run_sft_microbatch_train_step as sft_microbatch_train_step,
)
from transformers import Adafactor
from tqdm import tqdm



# ====================== SFT Data Tils ======================
#if args=true,use reward_fn filter the data can get right answer
#return filter how many examples
def filter_correct_sft_samples(
    data_path: str, 
    out_path: str,
    reward_fn: Callable[[str, Any], Dict[str, float]],
)->Dict[str,Any]:
    # MATH/sft.jsonl
    kept = []
    total = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)#Dict[str,Any]
            total += 1
            gt = get_ground_truth(example) 
            resp = example.get("response") #这里可以考虑也做成函数
            scores = reward_fn(resp, gt)
            if float(scores.get("answer_reward", 0.0)) >= 1.0:
                kept.append(example)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for example in kept:
            w.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    return {"filtered/kept": len(kept), "filtered/total": total}

# ====================== Data Set/Loader ======================
class SFTDataset(Dataset):
    def __init__(self, path: str, limit: int = 0, seed: int = 0):
    ## MATH/sft.jsonl or runs/sft_experiment/filtered_sft.jsonl
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        if limit and limit > 0:
            rnd = random.Random(seed)
            rnd.shuffle(self.data)
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    #ex : "prompt"..."response"..."ground_truth"
    def __getitem__(self, idx):
        ex = self.data[idx]
        return ex.get("prompt"), ex.get("response"), ex 
#tuple from DataSet.GetItem ==> token_ids,labels,response_mask
def collate_fn(batch, tokenizer):
    prompts = [x[0] for x in batch]
    outputs = [x[1] for x in batch]

    toks = tokenize_prompt_and_output(prompts, outputs, tokenizer)
    return toks
# get prompts and gts from validation.jsonl,prompts should pack with template
def build_math_val_prompts_and_gts(val_path: str, prompt_path: str, val_num: int = 0):
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    
    # get {quesion}...{answer}... from validation.jsonl
    examples = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    if val_num and val_num > 0:
        examples = examples[:val_num]

    prompts, gts = [], []
    for example in examples:#Dict[str,Any] 
        q = get_question(example)
        gt = get_ground_truth(example) 
        prompts.append(prompt_template.format(question=q))
        gts.append(gt)

    return prompts, gts 

# ====================== policy/llm ======================
def load_policy_into_vllm_instance(policy, llm: LLM):
    state_dict = policy.state_dict()

    def _load_weights(worker):
        model = worker.get_model()
        model.load_weights(state_dict.items())
        return True
    # main thread send func to sub thread
    llm.llm_engine.collective_rpc(_load_weights)
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_path = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    #patch : runtime replace internal func return val ,vllm skip memory check and set gpu num
    # vllm>=0.10.0 removed the `device` kwarg; device is auto-detected from CUDA
    with world_size_path, profiling_patch:
        return LLM(
            model=model_id,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def main():
    ap = argparse.ArgumentParser()
    # ====================== paths ======================
    ap.add_argument("--model_path", default="cs336_alignment/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--sft_path", default="data/MATH/sft.jsonl")
    ap.add_argument("--val_path", default="data/MATH/validation.jsonl")
    ap.add_argument("--prompt_path", default="cs336_alignment/prompts/r1_zero.prompt")
    ap.add_argument("--out_path", default="runs/sft_experiment")
    # ====================== device ======================
    ap.add_argument("--train_device", default="cuda:0")
    ap.add_argument("--vllm_device", default="cuda:0")
    # ====================== llm sample paras ======================
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    # ====================== train paras ======================
    ap.add_argument("--train_samples", type=int, default=500, help="0 means full dataset")
    ap.add_argument("--filter_correct", action="store_true",help="true means filter sft.jsonl")


    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0) 
    ap.add_argument("--micro_batch_size", type=int, default=2)
    ap.add_argument("--grad_acc_steps", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=2000)

    ap.add_argument("--eval_interval", type=int, default=0)
    ap.add_argument("--val_max_examples", type=int, default=500)
    ap.add_argument("--disable_eval", action="store_true")

    args = ap.parse_args() 

    #run_dir = samples(1000/full)_(filtered/all)
    run_dir = Path(args.out_path) / f"samples{args.train_samples or 'full'}_{'filtered' if args.filter_correct else 'all'}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- tee stdout/stderr + logging to console.log ---
    class _TeeWriter:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    console_log = open(run_dir / "console.log", "w", encoding="utf-8")
    sys.stdout = _TeeWriter(sys.__stdout__, console_log)
    sys.stderr = _TeeWriter(sys.__stderr__, console_log)
    logging.getLogger().addHandler(logging.StreamHandler(console_log))

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
            # keep terminal readable
            if "msg" in event:
                print(event["msg"])
            else:
                print(payload)    

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": args.train_device},
    )
    policy.train()

    opt_step = 0  # counts optimizer updates ,  "step % grad_acc_steps==0" then backward the grad
    step = 0

    #llm = init_vllm(args.model_path, device=args.vllm_device, seed=args.seed)
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    #get data from validation.jsonl
    eval_prompts, eval_gts = build_math_val_prompts_and_gts(
        args.val_path, args.prompt_path, val_num=args.val_max_examples
    )
    #  filter sft dataset ,data path will be replace to filtered_sft.jsonl
    data_path = args.sft_path
    if args.filter_correct:
        filtered_path = str(Path(args.out_path) / "filtered_sft.jsonl")
        stats = filter_correct_sft_samples(args.sft_path, filtered_path,r1_zero_reward_fn)
        log_event({"type": "filter_stats", "stats": stats, "msg": f"Filter stats: {stats}"})
        data_path = filtered_path

    dataset = SFTDataset(data_path, limit=args.train_samples, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        drop_last=True,
    )

    #opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    opt = Adafactor(
        policy.parameters(),
        lr=args.lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    opt.zero_grad(set_to_none=True)

    pbar = tqdm(total=args.max_steps, desc="SFT training", unit="step")
    for epoch in range(10_000_000):
        for batch in loader:
            step += 1
            pbar.update(1)
            # in dataLoader.collate_fn , example ==> [input_ids....]
            input_ids = batch["input_ids"].to(args.train_device)
            labels = batch["labels"].to(args.train_device)
            response_mask = batch["response_mask"].to(args.train_device)

            # get per-token log_probs (B, T)
            out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            policy_log_probs = out["log_probs"] 

            # loss  divide by grad_acc_steps ，SFT loss has no advantage
            # -nll = (log_probs*mask).sum(dim = 1),so shape (Batch_size,)
            # loss = -nll.mean , so loss only one val for whole microBatch
            loss, meta = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.grad_acc_steps,
                normalize_constant=1.0,
            )

            if step % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                pbar.set_postfix(loss=f"{float(loss.detach()):.4f}", opt_step=opt_step)
                if opt_step % 10 == 0:
                    log_event({"type": "train_loss", "loss": float(loss.detach())}, also_print=False)
            
            need_eval = ( not args.disable_eval
                and (
                    (args.eval_interval > 0 and step % args.eval_interval == 0)
                    or step == args.max_steps
                )
            )
            if need_eval:
                # --- temp vLLM for eval ---
                policy.cpu()
                torch.cuda.empty_cache()
                llm = init_vllm(
                    args.model_path, 
                    device=args.vllm_device, 
                    seed=args.seed, 
                    gpu_memory_utilization=0.50
                    )
                # --- start policy eval ---
                policy.eval()
                with torch.no_grad():
                    #llm already init,we only need to get paras from policy,then eval
                    load_policy_into_vllm_instance(policy, llm)
                    rows = evaluate_vllm(
                        vllm_model=llm,
                        reward_fn=r1_zero_reward_fn,
                        prompts=eval_prompts,
                        ground_truths=eval_gts,
                        sample_params=eval_sampling_params,
                        request_batch_size=64,
                    )
                n = len(rows)
                eval_acc = sum(r.answer_reward for r in rows) / n if n else 0.0
                eval_format = sum(r.format_reward for r in rows) / n if n else 0.0
                eval_reward = sum(r.reward for r in rows) / n if n else 0.0
                eval_metadata = {
                    "eval/accuracy": eval_acc,
                    "eval/format_rate": eval_format,
                    "eval/avg_reward": eval_reward,
                    "eval/n": n,                    
                }
                log_event({"type": "eval_metadata", "loss": float(loss.detach()), "metadata": eval_metadata,
                        "msg": f"[step={step}] loss={float(loss.detach()):.4f} {eval_metadata}"})
                # --- del temp vllm ---
                del llm
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                policy.to(args.train_device)
                policy.train()
            
            if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break
    pbar.close()
    # save
    policy.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    log_event({"type": "save", "out_dir": str(run_dir), "msg": f"Saved: {run_dir}"})

if __name__ == "__main__":
    main()
