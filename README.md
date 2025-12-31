# Training-Benchmark

### Setup

```bash
# Clone with submodules
git clone git@github.com:Xiaoming-AMD/Training-Benchmark.git
cd Training-Benchmark
git submodule update --init --recursive

# Set up pre-commit hooks (optional, for local linting/formatting)
pip install pre-commit
pre-commit install
```

The main training engine `Primus` is checked out as a git submodule under:

- `third_party/Primus`

## MI355 Pretrain & Perf Scripts

All MI355-related scripts are under `scripts/MI355` and use `run_pretrain_mi355x.sh` as the unified entry.

### 1. Single pretrain run

Example: run a single Qwen3 235B A22B pretrain job:

```bash
cd scripts/MI355

export MODEL_NAME=qwen3_235B_A22B
export NNODES=4          # number of nodes
export MBS=8             # micro batch size per GPU
export GBS=256           # global batch size
export TP=1              # tensor parallel size
export PP=4              # pipeline parallel size
export EP=8              # expert parallel size
export VPP=3             # virtual pipeline stages (optional)
export PIPELINE_LAYOUT="Et*7|(t*8|)*10,t*7,L"  # optional layout

bash run_pretrain_mi355x.sh
```

If `PIPELINE_LAYOUT` is set, it will be passed as `--pipeline_model_parallel_layout`. If it is not set and `VPP>1`, then `--num_virtual_stages_per_pipeline_rank` will be used instead.

### 2. Qwen3 235B perf sweep

To run the predefined Qwen3 235B performance sweep:

```bash
cd scripts/MI355
bash perf_test_qwen3_235b_a22b.sh
```

This script:
- Exports common perf flags (`MANUAL_GC`, `NUMA_BINDING`, `ENABLE_SYNC_FREE_MOE`, `ENABLE_TURBO_DEEPEP`)
- Iterates over a list of `(NNODES, MBS, GBS, PP, EP, VPP, PIPELINE_LAYOUT, TRAIN_ITERS)` configs
- Calls `run_pretrain_mi355x.sh` for each config
