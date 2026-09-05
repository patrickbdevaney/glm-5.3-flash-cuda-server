#!/usr/bin/env bash
# Build the kernels, the server, and every gate. Gates are the deliverable as much as the engine
# is: a kernel that is not gated against transformers on real weights is not finished (CLAUDE.md §2).
set -e; cd "$(dirname "$0")/.."
mkdir -p build
ARCH="-gencode arch=compute_110a,code=sm_110a"
K="kernels/kda.cu kernels/layer.cu kernels/moe.cu kernels/mla.cu kernels/gemv.cu"
E="src/engine.cu $K"

# CPU-only gates. These need no GPU and no checkpoint weights, so they run anywhere and are the
# first thing to check when something looks wrong at the text level rather than the tensor level.
for g in gate_tokenizer gate_encoding gate_sample gate_stream gate_api; do
  g++ -O2 -std=c++17 -I include tests/$g.cpp -o build/$g && echo "built build/$g"
done

for g in gate_kda gate_moe gate_layer gate_mla; do
  nvcc -O2 -std=c++17 $ARCH -I include tests/$g.cu $K -o build/$g && echo "built build/$g"
done
# the indexer needs no engine and no checkpoint shards: its oracle dumps its own inputs
nvcc -O2 -std=c++17 $ARCH -I include tests/gate_indexer.cu kernels/indexer.cu kernels/gemv.cu \
     -o build/gate_indexer && echo "built build/gate_indexer"
# gates that drive the whole engine
for g in gate_stack gate_batch; do
  nvcc -O2 -std=c++17 $ARCH -I include tests/$g.cu $E -o build/$g && echo "built build/$g"
done
for t in bench_kda; do
  nvcc -O2 -std=c++17 $ARCH -I include tools/$t.cu $K -o build/$t && echo "built build/$t"
done
nvcc -O2 -std=c++17 $ARCH -I include tools/bw_probe.cu -o build/bw_probe && echo "built build/bw_probe"
nvcc -O2 -std=c++17 $ARCH -I include tools/bench_batch.cu $E -o build/bench_batch && echo "built build/bench_batch"

# The server. -pthread for httplib's thread pool; the UI, API shaping and tokenizer are all headers.
nvcc -O2 -std=c++17 $ARCH -I include -Xcompiler -pthread \
     server/server.cpp $E -o build/glm5-server && echo "built build/glm5-server"
