#!/usr/bin/env bash
# Build the kernels and every gate. Gates are the deliverable as much as the engine is: a kernel
# that is not gated against transformers on real weights is not finished (CLAUDE.md §2).
set -e; cd "$(dirname "$0")/.."
mkdir -p build
ARCH="-gencode arch=compute_110a,code=sm_110a"
K="kernels/kda.cu kernels/layer.cu kernels/moe.cu kernels/mla.cu kernels/gemv.cu"
E="src/engine.cu $K"

for g in gate_kda gate_moe gate_layer gate_mla; do
  nvcc -O2 -std=c++17 $ARCH -I include tests/$g.cu $K -o build/$g && echo "built build/$g"
done
# gates that drive the whole engine
for g in gate_stack; do
  nvcc -O2 -std=c++17 $ARCH -I include tests/$g.cu $E -o build/$g && echo "built build/$g"
done
for t in bench_kda; do
  nvcc -O2 -std=c++17 $ARCH -I include tools/$t.cu $K -o build/$t && echo "built build/$t"
done
nvcc -O2 -std=c++17 $ARCH -I include tools/bw_probe.cu -o build/bw_probe && echo "built build/bw_probe"
