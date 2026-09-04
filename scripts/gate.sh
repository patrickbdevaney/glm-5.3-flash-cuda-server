#!/usr/bin/env bash
# Run every gate. Exits non-zero if any fails. Oracles must exist first:
#   cd ~/glm-5.3-reap
#   for g in kda moe layer; do ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_$g.py; done
set -u; cd "$(dirname "$0")/.."
rc=0
for g in gate_kda gate_moe gate_layer; do
  [ -x build/$g ] || { echo "MISSING build/$g - run scripts/build.sh"; rc=1; continue; }
  echo "=== $g ==="
  ./build/$g || rc=1
  echo
done
[ $rc -eq 0 ] && echo "ALL GATES PASSED" || echo "GATES FAILED"
exit $rc
