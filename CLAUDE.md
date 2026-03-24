# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPR (Principle-based Process Reward) is a reinforcement learning framework for training LLM search agents. It integrates a Principle-based Process Reward Model (PPRM) with Reward Normalization (ReNorm) on top of a PPO training loop. Built on veRL (Volcano Engine RL) and targets multi-turn search-based QA tasks (NQ, TriviaQA, HotpotQA, etc.).

## Setup & Installation

```bash
conda create -n rl_search python=3.10
conda activate rl_search
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb
```

## Key Commands

**Training (requires retrieval server + PPRM server running):**
```bash
# Launch retrieval server (Terminal 1)
bash retrieval_launch.sh

# Launch PPRM reward model via SGLang (Terminal 2)
bash pprm_launch.sh start pprm_3b_data 30000

# Run PPO training with PPR
python3 -m verl.trainer.main_ppo data.train_files=... actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct ...
```

**Example training scripts:** `examples/train_3b.sh`, `examples/train_7b.sh`

**Inference:** `python infer.py`

**Configuration:** Hydra-based. All config overrides via CLI: `param.nested.key=value`. Config schemas in `verl/trainer/config/`.

## Architecture

### Training Loop (`verl/trainer/main_ppo.py`)
Entry point. Defines `RewardManager` which orchestrates outcome + process reward scoring, then hands off to `RayPPOTrainer` for distributed PPO.

### Core PPO (`verl/trainer/ppo/`)
- `ray_trainer.py` — Ray-based distributed trainer orchestrating actor, critic, ref, and rollout workers
- `core_algos.py` — PPO loss, GAE advantage estimation, KL control

### Workers (`verl/workers/`)
All workers are Ray actors following a `Worker` base class with `execute()` method:
- `actor/` — Policy model, gradient updates via FSDP
- `critic/` — Value function for advantage estimation
- `rollout/vllm_rollout/` — Trajectory generation using vLLM
- `reward_model/` — Reward model worker interface
- `fsdp_workers.py` / `megatron_workers.py` — Distributed parallelism strategies

### Reward System (key innovation)
- **Outcome reward** (`verl/utils/reward_score/qa_em.py`): Exact-match scoring against golden answers
- **Process reward** (`verl/utils/reward_score/process_reward.py`): PPRM scores each step via async OpenAI-compatible API calls to an SGLang-served model. Uses `asyncio` with configurable concurrency (~800 concurrent requests)
- **ReNorm**: Combined reward normalization formula `r̂ = rp + ro - μ` (μ ≈ 1) to prevent outcome reward dilution in long trajectories. Configured via `process_reward_config.renorm=true` and `process_reward_config.renorm_beta`

### Data Protocol (`verl/protocol.py`)
`DataProto` is the unified data container passed between all workers. Wraps a dict of tensors + metadata.

### Search Integration (`search_utils/`)
- `search/retrieval_server.py` — Pyserini + FAISS retrieval backend
- `llm_agent/generation.py` — Multi-turn LLM generation manager

### Trajectory Format
Multi-turn search trajectories use XML-like tags:
```
<think>reasoning</think>
<search>query</search>
<information>retrieved results</information>
<answer>final response</answer>
```

## Supported Datasets
NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle — mapped in `_select_rm_score_fn()` in `main_ppo.py`.

## Dependencies
- Python 3.10, PyTorch 2.4, vLLM ≤0.6.3, transformers <4.48
- Ray for distributed orchestration, FSDP for model parallelism
- SGLang for serving the PPRM reward model
- WandB for experiment logging
