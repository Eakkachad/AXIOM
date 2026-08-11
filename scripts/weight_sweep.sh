#!/usr/bin/env bash
# T1.8a coordinate-ascent weight sweep driver.
# Usage: ./scripts/weight_sweep.sh <configs.json>  (each config = array of env pairs)
# Runs up to 8 full benches in parallel, prints config -> candidate/substring/recall.
set -u
DATA_QA="data/triviaqa/qa/verified-wikipedia-dev.json"
DATA_EV="data/triviaqa/evidence/wikipedia"
BIN="./target/release/triviaqa-bench"

run_one() {
  local cfg="$1"
  local label="$2"
  local out
  out=$(env $cfg "$BIN" "$DATA_QA" - "$DATA_EV" 2>/dev/null)
  local cand sub rec
  cand=$(echo "$out" | grep candidate | awk '{print $2}')
  sub=$(echo "$out" | grep substring | awk '{print $2}')
  rec=$(echo "$out" | grep 'answer_entity_recall' | awk '{print $2}')
  echo "RESULT $label cand=$cand sub=$sub rec=$rec cfg=$cfg"
}

# Run given (label,cfg) pairs, up to 8 at a time.
N=0
for entry in "$@"; do
  label="${entry%%|*}"
  cfg="${entry#*|}"
  run_one "$cfg" "$label" &
  N=$((N+1))
  if [ $N -ge 8 ]; then
    wait
    N=0
  fi
done
wait
