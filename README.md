# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Recreate This Environment On Another Machine

If you want to restore this repository on a different device, the shortest path is:

1. Clone your own copy of the repository.
2. Make sure the machine has Python 3.11 or 3.12 and `uv` installed.
3. Restore any local model checkpoints or other large artifacts you used for experiments. This repository does not keep generated model weights in Git, so anything under `runs/`, `checkpoints/`, `models/`, or similar experiment output directories must be copied over or downloaded again.
4. Install the Python environment:

```sh
uv sync --no-install-package flash-attn
uv sync
```

5. Run the tests or your scripts through `uv`:

```sh
uv run pytest
uv run python sft_experiment.py
```

If you use a GPU workflow, make sure the new machine has a compatible CUDA setup for `torch`, `flash-attn`, and `vllm`.

## Commit And Push Changes

After you modify code, the normal workflow is:

```sh
git status
git add <files-you-changed>
git commit -m "Describe the change"
git push
```

If this repository is tied to your personal GitHub account, make sure the remote points at your own repo before pushing:

```sh
git remote -v
git remote set-url origin git@github.com:your-username/your-repo.git
```

For larger changes, run the test suite before committing:

```sh
uv run pytest
```

