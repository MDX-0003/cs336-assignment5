from vllm import LLM,SamplingParams
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft_utils import (
    evaluate_vllm,
    get_ground_truth,
    get_question,
    load_jsonl,
    sample_examples,
    summarize,
    write_jsonl,
)

def main():
    ap = argparse.ArgumentParser()
    # ====================== paths ======================
    ap.add_argument("--model_path", default="data/a5-alignment/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--data_path", default="data/MATH/validation.jsonl")
    ap.add_argument("--prompt_path", default="cs336_alignment/prompts/r1_zero.prompt")
    ap.add_argument("--out_dir", default="runs/math_baseline")
    # ====================== llm sample paras ======================
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    # ====================== baseline run paras ======================
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="how many question run in baseline , 0 means no limit")
    args = ap.parse_args() 

    #prompt_template = ...{question}...
    prompt_template = Path(args.prompt_path).read_text()

    #examples = list[Dict[str,Any]]  {"qustion"..."answer"...}  
    examples = list(load_jsonl(args.data_path))
    if args.limit and args.limit > 0:
        examples = examples[: args.limit] 
    num_questions = len(examples)
    num_samples = args.limit if args.limit>0 else num_questions
    print(f"current sampling question :{num_samples}/{num_questions}")
     
    prompts, gts = [], []
    for example in examples:
        problem_str = get_question(example)
        answer_any = get_ground_truth(example)
        
        prompt = prompt_template.format(question=problem_str)

        prompts.append(prompt)
        gts.append(answer_any)

    sample_params = SamplingParams(
        temperature = args.temperature,
        top_p = args.top_p,
        max_tokens = args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output = True,
    )
    llm = LLM(model = args.model_path)
    eval_rows = evaluate_vllm(
            llm,
            r1_zero_reward_fn,
            prompts,
            gts,
            sample_params,
            args.batch_size
        )
    # eval_rows : List[EvalRow]
    # the format "eval Row" will not enter output jsonl
    # should summerize first

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # write out  per-example
    write_jsonl(str(out_dir / "predictions.jsonl"), eval_rows)
    # summary json
    summary = summarize(eval_rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # sample_examples()->List[EvalRow],pick first x EvalRow match "category"
    # r = EvalRow,asdict turn to Dict[str,Any]
    log_sample_num = 10
    samples = {
        "F1A1": [asdict(r) for r in sample_examples(eval_rows, "F1A1", log_sample_num)],
        "F1A0": [asdict(r) for r in sample_examples(eval_rows, "F1A0", log_sample_num)],
        "F0A0": [asdict(r) for r in sample_examples(eval_rows, "F0A0", log_sample_num)],
    }
    with open(out_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print("Saved to:", out_dir)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
