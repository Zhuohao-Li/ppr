# Hybrid Reward Normalization for Non-Verifiable Search Process Supervision

## 📖 Overview
<p align="center">
  <img src="assets/overview.png" alt="Overview" width="800">
</p>


This is a reinforcement learning framework that integrates principle-based process rewards and reward normalization to achieve stable and effective training of LLM agents in search task.

## Links

- [Installation](#installation)
- [Quick start](#quick-start)
- [Performance](#performance)
- [Acknowledge](#acknowledge)

## Installation

#### Environment

```bash
conda create -n rl_search python=3.10
conda activate rl_search
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb

# Local retriever env
pip install pyserini
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# sglang for reward model serving
# We recommend create a new environment with torch>=2.6 to install sglang, as using current environment may have package conflicts.
pip install sglang[all]
```


## Quick start

Train a 3B search LLM with process reward model on NQ dataset with e5 as the retriever and wikipedia as the corpus.

(1) Download the indexing and corpus.
```bash
save_path=/the/path/to/save
python scripts/download_corpus.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) Process the NQ dataset.
```bash
python scripts/data_process.sh
```

(3) Download the process reward models.
```bash
# Process reward model with 3B training data
huggingface-cli download --resume-download anonymous/PRM_3b_data --local-dir PRM_3b_data
```

(3) Launch a local retrieval server.
```bash
bash retrieval_launch.sh
```

(4) Run RL training with process reward model with Qwen2.5-3B-Instruct.
```bash
conda activate rl_search
bash examples/train_3b.sh
```

## Performance
#### Main Results
<p align="center">
  <img src="assets/main_result.png" alt="" width="800">
</p>

#### Case Study
<p align="center">
  <img src="assets/case_study.png" alt="" width="800">
</p>

## Acknowledge

The implementation of this project is built upon [veRL](https://github.com/volcengine/verl) [Search-R1](https://github.com/PeterGriffinJin/Search-R1/tree/main) and [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main).
We deeply appreciate these teams for their contributions to open-source research and development.
