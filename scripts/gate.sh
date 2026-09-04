#!/usr/bin/env bash
# Run every gate. Exits non-zero if any fails. Oracles must exist first:
#   cd ~/glm-5.3-reap
#   for g in kda moe layer mla stack; do ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_$g.py; done
set -u; cd "$(dirname "$0")/.."
rc=0
for g in gate_kda gate_moe gate_layer gate_mla gate_stack; do
  [ -x build/$g ] || { echo "MISSING build/$g - run scripts/build.sh"; rc=1; continue; }
  echo "=== $g ==="
  # keep stderr: a gate that dies on a CUDA allocation under contention must say so, not vanish
  ./build/$g 2>&1 || rc=1
  echo
done
[ $rc -eq 0 ] && echo "ALL GATES PASSED" || echo "GATES FAILED"
exit $rc
