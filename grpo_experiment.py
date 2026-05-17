import os

# vllm 0.10.0 uses msgpack by default; collective_rpc with callable requires pickle fallback
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
from pprint import pprint
from vllm import LLM,SamplingParams
import argparse
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
    run_compute_group_normalized_rewards as compute_group_normalized_rewards,
    run_grpo_microbatch_train_step as grpo_microbatch_train_step,
    run_masked_mean as masked_mean
)
from transformers import Adafactor
from tqdm import tqdm
from typing import Literal
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

# ====================== jsonl load ======================
def load_jsonl(path,
    prmopt_template,
    limit = 0,
    seed = 0)->tuple[List[str], List[str]]:
#return [prompts,gts],use for load train/val data
    with open(path,encoding = "utf-8") as f:
        data = [
            json.loads(line)for line in f
        ]
    if limit >0:
        data = data[:limit]
    rnd = random.Random(seed)
    rnd.shuffle(data)
    #data = Dict[str,Any] "question":...,"answer":...
    questions = []
    gts = []
    for line in data:
        question = get_question(line)
        question = prmopt_template.format(question = question)
        gt = get_ground_truth(line)

        questions.append(question)
        gts.append(gt)
    return questions,gts

# ====================== grpo utls ======================
LossType = Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]

def main():
    ap = argparse.ArgumentParser()
    # ====================== paths ======================
    ap.add_argument("--model_path", default="cs336_alignment/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--sft_path", default="data/MATH/sft.jsonl")
    ap.add_argument("--train_path", default="data/MATH/train.jsonl")
    ap.add_argument("--val_path", default="data/MATH/validation.jsonl")
    ap.add_argument("--prompt_path", default="cs336_alignment/prompts/r1_zero.prompt")
    ap.add_argument("--out_path", default="runs/grpo_experiment")
    # ====================== device ======================
    ap.add_argument("--train_device", default="cuda:0")
    ap.add_argument("--vllm_device", default="cuda:0")
    # ====================== llm sample paras ======================
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    # ====================== train paras ======================
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0) 
    ap.add_argument("--train_samples", type=int, default=5, help="0 means full dataset")
    ap.add_argument("--val_samples", type=int, default=5, help="0 means full dataset")
    ap.add_argument("--filter_correct", action="store_true",help="true means filter sft.jsonl")
    # ====================== rollout paras ======================
    ap.add_argument("--sample_per_rollout", default=4,help="how many times should one question be asked")
    ap.add_argument("--group_size", default=8,help="how many question in one train bacth")
    #rollout_size = sample_per_rollout*group_size
    ap.add_argument("--advantage_std", action="store_true")
    ap.add_argument("--cliprange", default=0.2)
    ap.add_argument("--loss_type", type=str, 
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip",],
        default="no_baseline")

    
    ap.add_argument("--micro_batch_size", type=int, default=2)
    ap.add_argument("--grad_acc_steps", type=int, default=16)
    #train_size=micro_batch_size*grad_acc_steps
    ap.add_argument("--max_steps", type=int, default=2000)

    ap.add_argument("--eval_interval", type=int, default=0)
    ap.add_argument("--val_max_examples", type=int, default=500)
    ap.add_argument("--disable_eval", action="store_true")

    args = ap.parse_args() 

    run_dir = Path(args.out_path) / f"samples{args.train_samples or 'full'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    rollout_size = args.sample_per_rollout * args.group_size
    # rollout = ask times * quesions num
    train_size=args.micro_batch_size*args.grad_acc_steps
    loss_type = args.loss_type
    # -------- load data,prompt should templated --------
    prompt_template = Path(args.prompt_path).read_text(encoding="utf-8")
    train_questions,train_gts=load_jsonl(args.train_path,
               prompt_template,
               args.train_samples,#limit
               0)
    val_question,val_gts=load_jsonl(args.val_path,
               prompt_template,
               args.val_samples,#limit
               0)
    
    # -------- load model --------
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
            gpu_memory_utilization=0.50
        )
    load_policy_into_vllm_instance(policy, policy_vllm)
    # -------- load optimizer --------
    #opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    opt = Adafactor(
        policy.parameters(),
        lr=args.lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    opt.zero_grad(set_to_none=True)

    rollout_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    pbar = tqdm(total=args.max_steps, desc="GRPO training", unit="step")
    for step in range(args.max_steps):
        # ========== 1) sample prompts ==========
        batch_idxs = [random.randrange(0, len(args.train_samples)) for _ in range(rollout_size)]
        batch_prompts = [train_questions[i] for i in batch_idxs]# shape[rollout_size]
        batch_gts = [train_gts[i] for i in batch_idxs]

        # ========== 2) policy rollout ==========
        outputs = policy_vllm.generate(batch_prompts, rollout_sampling_params)
        rollout_prompts = []
        rollout_responses = []
        rollout_gts = []

        for request,gt in zip(outputs,batch_gts):
            # for one RequestOuput(generate() -> return value),outputs num can > 1
            prompt = request.prompt
            output  = request.outputs[0].text

            rollout_prompts.append(prompt)
            rollout_responses.append(output)
            rollout_gts.append(gt)
        # ========== 3) compute group-normalized rewards (advantages) ==========
        advantages, raw_rewards,metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=rollout_gts,
            group_size=rollout_size,
            advantage_eps=1e-6,
            normalize_by_std=args.advantage_std,
        )
        # ========== 4) tokenize prompt+response for scoring ==========
        toks = tokenize_prompt_and_output(
            prompt_strs=rollout_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        # toks: input_ids, labels, response_mask
        input_ids = toks["input_ids"].cuda()
        labels = toks["labels"].cuda()
        response_mask = toks["response_mask"].cuda()

        # ========== 5) optional , get old policy log probs ==========
        old_log_probs=None
        if loss_type == LossType.grpo_clip:
            with torch.inference_mode():
                old_policy_outs = get_response_log_probs(
                    policy,
                    input_ids,
                    labels,
                    True
                )
                old_log_probs = old_policy_outs["log_probs"].detach() # [B,T]
                old_log_probs.requires_grad_(False)
        # ========== 6) random shuffle Tensor use to cal loss ==========
        perm = torch.randperm(rollout_size, device=input_ids.device)
        advantages_gpu = advantages.cuda()[perm].unsqueeze(-1)      # (B, 1)
        raw_rewards_gpu = raw_rewards.cuda()[perm].unsqueeze(-1)    # (B, 1)
        input_ids = input_ids[perm]
        labels = labels[perm]
        response_mask = response_mask[perm]
        if old_log_probs is not None:
            old_log_probs = old_log_probs.cuda()[perm]
        # ========== 7) train loop ==========
        policy.train()
        global_step = 0
        for epoch in range(10_000_000):
            for idx in range(train_size):
                start = idx * train_size
                end = start + train_size

                batch_input_ids = input_ids[start:end]
                batch_labels = labels[start:end]
                batch_response_mask = response_mask[start:end]
                batch_advantages = advantages_gpu[start:end]
                batch_raw_rewards = raw_rewards_gpu[start:end]
                batch_old_log_probs = old_log_probs[start:end] if old_log_probs is not None else None

                opt.zero_grad(set_to_none=True)

                # used for logging
                loss_accum = 0.0
                entropies = []
                # how many micro_batch to update grads
                for micro_batch_idx in range(args.grad_acc_steps):
                    mb_input_ids = batch_input_ids[start:end]
                    mb_labels = batch_labels[start:end]
                    mb_response_mask = batch_response_mask[start:end]
                    mb_advantages = batch_advantages[start:end]
                    mb_raw_rewards = batch_raw_rewards[start:end]
                    mb_old_log_probs = batch_old_log_probs[start:end] if batch_old_log_probs is not None else None

                    scored = get_response_log_probs(
                        model=policy,
                        input_ids=mb_input_ids,
                        labels=mb_labels,
                        return_token_entropy=True,
                    )
                    policy_log_probs = scored["log_probs"]          # (microB, T)
                    token_entropy = scored["token_entropy"]         # (microB, T)

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

                # prevent excessively large updates
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                opt.step()

                global_step += 1
                if args.eval_interval >0 and global_step % args.eval_interval == 0:
                    save_dir = log_path / f"ckpt_step_{global_step}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                #after grads update, store new policy to vllm
                load_policy_into_vllm_instance(policy, policy_vllm)

if __name__ == "__main__":
    main()