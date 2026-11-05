# Memory-Aware Agent Edge Serving — Phase 1

Reproducibility materials for a Phase 1 study of concurrent LLM-agent execution, FIFO admission, context growth, and KV-cache reuse on a memory-constrained consumer GPU.

The repository contains a real, multi-step **Click code-investigation agent** and controlled **E1–E9** characterization experiments. It is an experimental baseline and measurement suite, **not** an implemented memory-aware scheduler or GPU KV-offloading system.

## 1. Workload and evaluation

- **Application workload:** 30 code-investigation tasks against the [Pallets Click](https://github.com/pallets/click) Python repository. The agent plans source/test searches, inspects source and tests, invokes focused `pytest`, and reports a source file, symbol, and related tests. It does **not** patch the repository.
- **Concurrency workload:** a fixed subset of eight tasks, submitted together to an explicit FIFO task queue. A fixed worker pool admits 1, 2, or 4 complete tasks, while a separate global semaphore permits at most two concurrent LLM requests.
- **Controlled workloads:** synthetic, reproducible prompt lengths, tool-like pauses, slot assignments, and context-growth patterns (E1–E9). These isolate serving effects and should not be interpreted as real coding-agent correctness tests.
- **Correctness:** `agent_executed`, reference implementation localization, and successful focused `pytest` are **different** measures. A passing focused test alone does not establish correct localization.

### Repository layout

```text
memory-aware-agent/
├── README.md
├── requirements.txt
├── llama-cpp-commit.txt
├── run_phase1_e6_budgets.sh
├── coding_agent_feasibility/
│   ├── coding_agent_correctness_5.py        # Click agent + grader
│   ├── run_memory_concurrency_fifo.py      # FIFO baseline
│   ├── profile_gpu_memory.py              # device-wide GPU sampler
│   ├── phase1_experiments.py              # E1–E3
│   ├── phase1_experiment_e4.py            # E4
│   ├── phase1_experiment_e5.py            # E5
│   ├── phase1_experiment_e6.py            # E6
│   ├── dynamic_gpu_feasibility.py         # E7
│   ├── cuda_vram_contender.py             # E7 helper
│   ├── phase1_experiment_e8.py            # E8
│   ├── phase1_context_growth.py           # E9
│   ├── cases_30.json
│   ├── cases_30_diverse_v7.json
│   └── test_*.py
└── results/                             # selected Phase 1 reference outputs
```

`coding-repos/click/`, `llama.cpp/`, Python virtual environments, model weights, and historical exploratory files are **not committed**. They are external inputs that should be prepared locally as described below.

## 2. Requirements and installation

The original measurements used an **NVIDIA GeForce RTX 5060 Laptop GPU (8 GB)** and a locally hosted **Qwen2.5-3B-Instruct Q4_K_M GGUF** model with `llama.cpp`. Similar hardware may reproduce the experiment *procedure*, but latency, GPU allocation, and success under capacity pressure can differ across server builds and GPUs.

Prerequisites:

- Linux / WSL2, Python **3.10+**, Git, an NVIDIA GPU and compatible driver for GPU experiments.
- `nvidia-smi`, `curl`, and an NVIDIA-enabled `llama-server` executable.
- `pytest` for the Click repository's real focused tests; the Phase 1 Python experiment harnesses themselves use the Python standard library.
- The `llama.cpp` build/commit recorded in `llama-cpp-commit.txt`. Use its recorded revision rather than assuming the latest upstream build behaves identically.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Prepare the pinned Click repository

The workload JSON files pin Click to commit `2247b35ea1c47c727d7a06e51fa280e12a863ff6`.

```bash
mkdir -p coding-repos
git clone https://github.com/pallets/click.git coding-repos/click
git -C coding-repos/click checkout 2247b35ea1c47c727d7a06e51fa280e12a863ff6
python -m pip install -e ./coding-repos/click
```

The runners check the Click Git revision and fail if it does not match the cases file. For comparable runs, keep the checkout unmodified. The actual Click tests execute with the Python interpreter running the agent; use this same environment for `pytest`.

### Prepare llama.cpp

Clone and build `llama.cpp` separately under `./llama.cpp`, using the revision documented in `llama-cpp-commit.txt` and a CUDA-enabled build. For example, after checking out that revision:

```bash
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j
```

The model is loaded via llama.cpp's Hugging Face `-hf` option; an initial download or a suitable local model cache is required. No model weights are included in this repository.

The following is the **common two-slot configuration**, used with shared logical KV budget `-c 8192` for the FIFO baseline and E1/E2/E4/E5/E7:

```bash
mkdir -p /tmp/llama-phase1-slots
./llama.cpp/build/bin/llama-server \
  -hf Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M \
  -ngl 99 -c 8192 -np 2 \
  --kv-unified --kv-unified-per-slot 4096 \
  --cache-ram 0 --slot-save-path /tmp/llama-phase1-slots \
  --metrics -cb --host 127.0.0.1 --port 8080
```

In a second terminal, activate the virtual environment and check:

```bash
curl -fsS http://127.0.0.1:8080/health
python -m unittest discover -s coding_agent_feasibility -p 'test_*.py' -v
```

The offline unit tests do not need a GPU server. For **E3, E6, and E9**, stop the running server and restart with the indicated `-c` value; keep `-np 2`, `--kv-unified-per-slot 4096`, and the other options unchanged. The experiments verify some server settings and will refuse mismatches. Run cache-interference experiments against an otherwise idle server: they erase slot caches between trials.

> **Output directories:** reference data already exists under `results/`. The commands below write fresh measurements under `repro_results/` to avoid overwriting the supplied results. Do not reuse an existing nonempty experiment output directory; several experiment drivers explicitly reject that.

## 3. Reproducing the Click workload and FIFO baseline

### Thirty-case code-investigation workload

With the server available at `http://127.0.0.1:8080/v1` and pinned Click installed:

```bash
python coding_agent_feasibility/coding_agent_correctness_5.py \
  --cases-file coding_agent_feasibility/cases_30_diverse_v7.json \
  --max-inflight-llm 2 --max-followups 1 \
  --out repro_results/coding_cases_30_diverse_v7.json
```

The saved report's selected workload result is under `results/coding_cases_30_diverse_v7_1.json`.

### Fixed-concurrency FIFO (eight identical cases)

Run each configuration on the same model/server build. All eight tasks are enqueued as a batch in dataset order:

```bash
mkdir -p repro_results
for c in 1 2 4; do
  python coding_agent_feasibility/profile_gpu_memory.py \
    --out "repro_results/fifo_gpu_c${c}.csv" --interval-ms 500 -- \
    python coding_agent_feasibility/run_memory_concurrency_fifo.py \
      --cases-file coding_agent_feasibility/cases_30_diverse_v7.json \
      --cases group_resolution cli_runner context_forward choice_error \
              context_invoke terminal_prompt runner_isolation file_type_convert \
      --max-tasks "$c" --max-inflight-llm 2 --max-followups 1 \
      --slots-out "repro_results/fifo_slots_c${c}.jsonl" \
      --out "repro_results/fifo_concurrency_c${c}.json"
done
```

Reference outputs: `results/fifo_concurrency_c{1,2,4}.json` and `results/fifo_gpu_c{1,2,4}.*`.

Useful fields in each FIFO JSON's `summary`: completed tasks/s, implementation localization correctness, end-to-end task-latency P95, FIFO task-queue P95, task-execution P95, and client-side LLM-semaphore queue P95. **Client LLM queue time is not llama-server internal queue time**, and GPU profiler MiB is whole-device allocation, not active KV occupancy.

## 4. Controlled experiments E1–E9

These experiments use native llama.cpp completion/slot endpoints, rather than the OpenAI-compatible `/v1` endpoint used by the Click agent. Synthetic tool delays model pauses; they are not real `grep` or `pytest` execution time.

| Study question | Experiments | Reference data |
| --- | --- | --- |
| Can a request fail because concurrent contexts exceed shared KV capacity? | E3, E6, E9 | `results/phase1_e3_c4096/`, `results/phase1_e6_*`, `results/context_growth_*` |
| What allows cached prefixes to survive a tool pause? | E1, E2, E6 idle-KV | `results/phase1_e1_c8192/`, `results/phase1_e2_c8192/`, `results/phase1_e6_*` |
| When should a paused agent keep or release a server slot? | E4, E5, E6 sweep | `results/phase1_e4_c8192/`, `results/phase1_e5_c8192/`, `results/phase1_e6_*` |
| Does logical cache erasure release physical GPU memory? | E7, E8 | `results/e7_f_vram_*.json`, `results/phase1_e8_physical_vram/` |

### E1 and E2 — Tool waiting and slot interference (`-c 8192`)

```bash
python coding_agent_feasibility/phase1_experiments.py \
  --experiment e1 --expect-ctx 8192 --waits 0 3 8 \
  --a-tokens 1200 --n-predict 64 \
  --out-dir repro_results/phase1_e1_c8192

python coding_agent_feasibility/phase1_experiments.py \
  --experiment e2 --expect-ctx 8192 \
  --a-tokens 1200 --b-short-tokens 1200 --b-long-tokens 2600 \
  --n-predict 64 --gap 3 \
  --out-dir repro_results/phase1_e2_c8192
```

### E3 — Concurrent vs. sequential contexts (`-c 4096`)

**Restart** the server with `-c 4096` before this run.

```bash
python coding_agent_feasibility/phase1_experiments.py \
  --experiment e3 --expect-ctx 4096 \
  --a-tokens 1200 --b-short-tokens 1200 --b-long-tokens 2600 \
  --n-predict 128 \
  --out-dir repro_results/phase1_e3_c4096
```

### E4 — Preserve versus reuse a paused agent's slot (`-c 8192`)

**Restart** the server with `-c 8192`:

```bash
python coding_agent_feasibility/phase1_experiment_e4.py \
  --expect-ctx 8192 --waits 0 3 8 \
  --mixes short_short short_long --repeats 3 \
  --a-tokens 1200 --short-tokens 1200 --long-tokens 2600 \
  --a-predict 64 --bc-predict 128 \
  --out-dir repro_results/phase1_e4_c8192
```

`Preserve` and `Reuse` are **client-side slot-placement choices**, not proof of explicit KV pinning or physical memory migration.

### E5 — TTL versus static retention (`-c 8192`)

```bash
python coding_agent_feasibility/phase1_experiment_e5.py \
  --expect-ctx 8192 --mixes short_short short_long \
  --wait-values 0 0.5 3 8 --episodes-per-mix 8 \
  --ttls 0.5 1.5 --seed 42 \
  --a-tokens 1200 --short-tokens 1200 --long-tokens 2600 \
  --a-predict 64 --bc-predict 128 \
  --out-dir repro_results/phase1_e5_c8192
```

### E6 — Sweep of shared KV budget and preservation policy

For **each** budget in `4096`, `6144`, and `8192`, restart the server with exactly that `-c` setting and run:

```bash
CTX=4096  # change to 6144, then 8192 after restarting the server
python coding_agent_feasibility/phase1_experiment_e6.py \
  --expect-ctx "$CTX" --mode both \
  --waits 0 3 --mixes short_long long_long \
  --b-profiles short long --a-tokens 1200 2200 \
  --short-tokens 1200 --long-tokens 2600 \
  --a-predict 64 --bc-predict 128 --repeats 2 \
  --sample-interval 0.5 \
  --out-dir "repro_results/phase1_e6_c${CTX}"
```

After all three budgets finish (no server required for aggregation):

```bash
python coding_agent_feasibility/phase1_experiment_e6.py \
  --aggregate repro_results/phase1_e6_c4096 \
              repro_results/phase1_e6_c6144 \
              repro_results/phase1_e6_c8192 \
  --out-dir repro_results/phase1_e6_comparison
```

The optional `run_phase1_e6_budgets.sh` starts/stops the server automatically, **but uses fixed `results/phase1_e6_*` paths**. It will refuse to overwrite the reference data already in this repository. The manual commands above avoid this conflict.

### E7 — Logical cache erasure and external GPU contender (`-c 8192`)

**Restart** the server with `-c 8192`. The contender requires a working NVIDIA CUDA driver; allocate only when enough physical VRAM is free:

```bash
python coding_agent_feasibility/dynamic_gpu_feasibility.py \
  --tokens 2200 --n-predict 64 --wait 2 --skip-contender \
  --out repro_results/e7_f_vram_baseline.json

python coding_agent_feasibility/dynamic_gpu_feasibility.py \
  --tokens 2200 --n-predict 64 --wait 2 \
  --contender-mib 512 --contender-hold 5 \
  --out repro_results/e7_f_vram_contender.json
```

### E8 — KV allocation across server startup budgets

**Stop** the existing server before E8. The E8 harness launches and stops its *own* server instances and needs `./llama.cpp/build/bin/llama-server` (or `--server-binary`):

```bash
python coding_agent_feasibility/phase1_experiment_e8.py \
  --budgets 4096 8192 16384 --per-slot 4096 \
  --tokens 2200 --predict 64 --wait 2 \
  --repeats 3 --samples 5 \
  --out-dir repro_results/phase1_e8_physical_vram
```

### E9 — Fixed versus growing context (`-c 4096`, then `-c 8192`)

Restart the server at each budget, keeping two slots and `--kv-unified-per-slot 4096`:

```bash
CTX=4096  # repeat with CTX=8192 after restarting the server
python coding_agent_feasibility/phase1_context_growth.py \
  --expect-ctx "$CTX" --mode both --repeats 3 \
  --targets 900 1800 2700 --wait 1 --n-predict 32 \
  --sample-interval 0.5 \
  --out-dir "repro_results/context_growth_c${CTX}"
```

Offline aggregation:

```bash
python coding_agent_feasibility/phase1_context_growth.py \
  --aggregate repro_results/context_growth_c4096 \
              repro_results/context_growth_c8192 \
  --out-dir repro_results/context_growth_combined
```

E9's `completed` counts mean that all three synthetic inference steps succeeded, **not** that the agent solved a coding task correctly.

## 5. Results and measurement caveats

The committed `results/` directory contains the selected reference CSV/JSON outputs corresponding to the Phase 1 report. For independent reruns, compare the new `repro_results/` output with the reference directories shown above. Run-to-run variance is expected; the experiments were performed on a single GPU and many policy comparisons have limited repetitions.

- `-c` is the llama.cpp **logical shared KV-token budget**, not a limit on physical GPU VRAM.
- `nvidia-smi` whole-device memory includes model weights, KV buffers, CUDA/runtime allocations, and any other GPU processes. Erasing a slot's prefix need not release physical VRAM.
- Request duration is end-to-end **client-observed request latency**, not exclusive GPU execution time.
- Successful synthetic model calls and successful Click-agent execution should not be conflated with correct implementation localization.
- The E4/E5 policy comparisons use synthetic tool waiting and fixed slot placement. They do not implement an online memory-aware scheduler or CPU/GPU swapping.
- Prompt tokenization, cache-hit counts, and context-capacity failures can depend on the llama.cpp commit and exact launch flags. Record these along with Python environment and GPU details for new measurements.

## 6. Attribution

- [Click](https://github.com/pallets/click): upstream repository used for code investigation; checked out locally at the pinned commit above.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): model-serving runtime, built separately at the revision documented in `llama-cpp-commit.txt`.
- [Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF): model checkpoint used in the experiments, not redistributed here.
