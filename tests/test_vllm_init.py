"""Minimal smoke test: verify vLLM can initialize with the Qwen2.5-Math-1.5B model."""
import torch
from vllm import LLM, SamplingParams


def test_vllm_basic_init():
    from pathlib import Path
    # Relative to PROJECT ROOT (where you run the script from), not this file
    model_path = str(Path(__file__).resolve().parent.parent / "cs336_alignment/models/Qwen2.5-Math-1.5B")
    print(f"torch: {torch.__version__} | cuda available: {torch.cuda.is_available()}")

    llm = LLM(
        model=model_path,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.50,
        max_model_len=512,
    )
    print("vLLM engine started OK")

    # Quick forward pass sanity check
    outputs = llm.generate(
        ["What is 2+2?"],
        SamplingParams(temperature=1.0, max_tokens=32),
    )
    print(f"Generated: {outputs[0].outputs[0].text!r}")

    del llm
    torch.cuda.empty_cache()
    print("PASS")


if __name__ == "__main__":
    test_vllm_basic_init()
