#!/usr/bin/env bash
# Launch the full 45-layer engine, detached and OOM-safe.
#
# Why the oom_score_adj line: a cudaMalloc on Thor comes out of the same DRAM as
# everything else, so an over-allocation is a SYSTEM OOM, not a CUDA error return.
# The kernel then picks a victim by score, and it picked the Claude Code session
# last time. Raising our own score to the maximum makes this process the victim
# instead, so a bad sizing costs a restart rather than the whole session.
set -u
echo 1000 > /proc/self/oom_score_adj
cd "$(dirname "$0")/.."
exec ./build/glm5-server "$@"
