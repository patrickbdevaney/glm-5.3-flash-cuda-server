#!/usr/bin/env bash
# Run every gate. Exits non-zero if any fails.
#
# The CPU gates need nothing but the checkpoint's tokenizer.json and the vectors in ref/, so they
# run on a busy box. The GPU gates need their oracles generated first:
#   cd ~/glm-5.3-reap
#   for g in kda moe layer mla stack; do ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_$g.py; done
# gate_batch is the exception: its reference is the engine's own sequential path, so it needs no
# oracle and runs whenever the weights fit.
set -u; cd "$(dirname "$0")/.."
rc=0
CPU="gate_tokenizer gate_encoding gate_sample gate_stream gate_api"
GPU="gate_kda gate_moe gate_layer gate_mla gate_stack gate_batch"
ONLY="${1:-all}"
case "$ONLY" in cpu) LIST="$CPU";; gpu) LIST="$GPU";; *) LIST="$CPU $GPU";; esac
for g in $LIST; do
  [ -x build/$g ] || { echo "MISSING build/$g - run scripts/build.sh"; rc=1; continue; }
  echo "=== $g ==="
  # keep stderr: a gate that dies on a CUDA allocation under contention must say so, not vanish
  ./build/$g 2>&1 || rc=1
  echo
done
[ $rc -eq 0 ] && echo "ALL GATES PASSED" || echo "GATES FAILED"
exit $rc
