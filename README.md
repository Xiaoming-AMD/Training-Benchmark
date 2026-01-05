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


## MI355 Pretrain & Perf Scripts

- All MI355-related scripts are under `scripts/MI355` and use `run_pretrain_mi355x.sh` as the unified entry.
- All training log files are stored under the path: third_party/Primus/output/date-$(date +%Y%m%d).

> **Note:**
> Before running the following scripts, please make sure to set the environment variables provided by AMD. These are required to pull the `tasimage` Docker images and access Huggingface tokenizers (replace with your own token if needed):
>
> ```bash
> export HF_TOKEN="your_hf_token"
> export DOCKER_LOGIN_USER="login_user"
> export DOCKER_LOGIN_KEY="login_key"
> ```


### 1. Qwen3 235B Benchmark

Run the predefined Qwen3 235B A22B performance sweep:

```bash
bash scripts/MI355/perf_test_qwen3_235b_a22b.sh
```

This script:
- Exports common perf flags (`MANUAL_GC`, `NUMA_BINDING`, `ENABLE_SYNC_FREE_MOE`, `ENABLE_TURBO_DEEPEP`)
- Iterates over a list of `(NNODES, MBS, GBS, PP, EP, PIPELINE_LAYOUT, TRAIN_ITERS)` configs
- Calls `run_pretrain_mi355x.sh` for each config

### 2. Qwen3 30B A3B Benchmark

```bash
bash scripts/MI355/perf_test_qwen3_30b_a3b.sh
```

This script:
- Sets `MODEL_NAME=qwen3_30B_A3B` and enables `NUMA_BINDING`, `ENABLE_SYNC_FREE_MOE`, `ENABLE_TURBO_DEEPEP`
- Sweeps `(NNODES, MBS, GBS, TP, PP, EP, CP, RECOMPUTE_LAYERS, TRAIN_ITERS, LOG_AVG_SKIP_ITERATIONS)`
- Calls `run_pretrain_mi355x.sh` for each config

### 3. Qwen3 8B Benchmark

```bash
bash scripts/MI355/perf_test_qwen3_8b.sh
```

This script:
- Sets `MODEL_NAME=qwen3_8B`
- Sweeps `(NNODES, MBS, GBS, TRAIN_ITERS)`
- Calls `run_pretrain_mi355x.sh` for each config

### 4. Llama3.1 8B Benchmark

```bash
bash scripts/MI355/perf_test_llama31_8b.sh
```

This script:
- Sets `MODEL_NAME=llama3.1_8B`
- Sweeps `(NNODES, MBS, GBS, TRAIN_ITERS)`
- Calls `run_pretrain_mi355x.sh` for each config

### 5. Llama3.1 70B Benchmark

```bash
bash scripts/MI355/perf_test_llama31_70b.sh
```

This script:
- Sets `MODEL_NAME=llama3.1_70B` or `MODEL_NAME=llama3.1_70B-Zero`
- Sweeps `(MODEL_NAME, NNODES, MBS, GBS, TP, PP, VPP, RECOMPUTE_LAYERS, TRAIN_ITERS)`
- Calls `run_pretrain_mi355x.sh` for each config
