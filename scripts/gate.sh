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
GPU="gate_kda gate_moe gate_layer gate_mla gate_indexer gate_nvfp4 gate_mla_sparse gate_vision gate_stack gate_batch"
ONLY="${1:-all}"
case "$ONLY" in cpu) LIST="$CPU";; gpu) LIST="$GPU";; *) LIST="$CPU $GPU";; esac
for g in $LIST; do
  [ -x build/$g ] || { echo "MISSING build/$g - run scripts/build.sh"; rc=1; continue; }
  echo "=== $g ==="
  # keep stderr: a gate that dies on a CUDA allocation under contention must say so, not vanish
  #
  # gate_stack compares the engine against a bf16/fp32 PyTorch oracle at cos >= 1-1e-6. NVFP4 is a
  # 4-bit format; it cannot meet that bound and should not be asked to. So the EXACTNESS gate runs
  # on bf16, where a failure still means a wiring bug, and the NVFP4 drift is REPORTED separately
  # below rather than folded into a pass/fail against weights the engine is no longer using.
  # gate_nvfp4 is what gates the overlay itself, against the tensors it was derived from.
  if [ "$g" = "gate_stack" ]; then GLM5_DENSE_NVFP4=0 ./build/$g 2>&1 || rc=1
  else ./build/$g 2>&1 || rc=1; fi
  echo
done
if [ -x build/gate_stack ] && [ "$ONLY" != "cpu" ]; then
  echo "=== gate_stack, NVFP4 dense weights (drift REPORT, not a pass/fail) ==="
  ./build/gate_stack 2>&1 | grep -E "step|cos" || true
  echo
fi
[ $rc -eq 0 ] && echo "ALL GATES PASSED" || echo "GATES FAILED"
exit $rc
