#!/usr/bin/env bash
# Optional all-budget runner. Refuses to interrupt a running E5 / llama-server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if pgrep -af llama-server >/dev/null; then
  echo 'ERROR: llama-server is already running. Finish E5 and stop that server first.' >&2
  exit 1
fi
mkdir -p results/phase1_e6_logs
SERVER_PID=''
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=''
  fi
}
trap cleanup EXIT INT TERM
for CTX in 4096 6144 8192; do
  DIR="results/phase1_e6_c${CTX}"
  if [[ -e "$DIR/sweep_trials.jsonl" || -e "$DIR/idle_trials.jsonl" ]]; then
    echo "ERROR: $DIR already contains E6 trials. Use a fresh run directory." >&2
    exit 1
  fi
  mkdir -p "/tmp/llama-e6-slots-${CTX}"
  echo "Starting server -c=${CTX}"
  ./llama.cpp/build/bin/llama-server \
    -hf Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M \
    -ngl 99 -c "$CTX" -np 2 \
    --kv-unified --kv-unified-per-slot 4096 \
    --cache-ram 0 --slot-save-path "/tmp/llama-e6-slots-${CTX}" \
    --metrics -cb --host 127.0.0.1 --port 8080 \
    >"results/phase1_e6_logs/server_c${CTX}.log" 2>&1 &
  SERVER_PID=$!
  READY=0
  for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "ERROR: llama-server -c=${CTX} exited unexpectedly. Check results/phase1_e6_logs/server_c${CTX}.log" >&2
      exit 1
    fi
    if curl -fsS --max-time 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
      READY=1
      break
    fi
    sleep 1
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: server -c=${CTX} never became healthy. Check its log." >&2
    exit 1
  fi
  python coding_agent_feasibility/phase1_experiment_e6.py \
    --expect-ctx "$CTX" --mode both \
    --waits 0 3 --mixes short_long long_long \
    --b-profiles short long --a-tokens 1200 2200 \
    --short-tokens 1200 --long-tokens 2600 \
    --a-predict 64 --bc-predict 128 --repeats 2 \
    --sample-interval 0.5 --out-dir "$DIR"
  cleanup
  # Ensure the port is no longer bound before starting the next budget.
  sleep 2
done
python coding_agent_feasibility/phase1_experiment_e6.py \
  --aggregate \
    results/phase1_e6_c4096 \
    results/phase1_e6_c6144 \
    results/phase1_e6_c8192 \
  --out-dir results/phase1_e6_comparison
printf 'E6 complete. See results/phase1_e6_comparison/\n'
