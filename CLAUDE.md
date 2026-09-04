# Operating rules for this repo

## 1. Detachment: if it does not need steering, it must not be tied to a session

Any long-running workload that does not depend on Claude Code to steer it **must be detached
before it starts** — benchmarks, eval batteries, trace capture, fine-tunes, profiling sweeps.

    nohup setsid <cmd> > <log> 2>&1 < /dev/null &      # minimum
    systemd --user transient unit                       # better; survives logout

`nohup setsid` is **not sufficient when the launcher is itself a systemd unit**: `setsid` changes
the session, not the cgroup, and `KillMode=control-group` kills the whole cgroup when the unit's
main process exits. Give each unattended stage its own transient unit.

**The converse is part of the rule.** Iterative kernel edit/build/gate loops, debugging, anything
where the next step depends on reading the last result — those are session-bound and should be.

**Verify parentage, not intent.** PPID must be 1 or `systemd --user`, TTY `?`. Audit what is
actually running, not what the script says.

Why: on 2026-08-17 a battery lost ~16 hours to a closed laptop lid. Stages launched with
`nohup setsid` survived the identical event on the same box at the same instant.

## 2. Every kernel is gated against a PyTorch oracle before it is believed

The rule inherited from `deepseek-v4-flash-0731-cuda` and `gemma-cuda-server`, which is why those
engines are trustworthy: **no kernel is "done" until a gate compares it to `transformers` output
on real checkpoint weights**, not on random tensors.

- `ref/gen_units.py` writes oracle tensors from the real model.
- `tests/gate_*.cu` load them and assert.
- Bit-exact where the op is exactly reproducible; `cosine >= 1 - 1e-6` / `max_rel <= 4e-3` where
  fp order differs.
- A gate that passes against a **dead engine or an absent file** is worse than no gate. Gates
  exit non-zero on missing inputs and never "pass" vacuously.

## 3. Measure on this box; inherit no numbers

Bandwidth, batch-cost curves, tok/s — all measured here (`tools/bw_probe.cu`, `tools/roofline.py`).
The one figure carried in from a sibling repo is 240 GB/s achievable / 212 contended, and it was
measured on **this** Thor. `ROOFLINE.md` says which numbers are measured and which are derived.

**`ncu`'s "Memory Throughput %" is not DRAM bandwidth utilisation on Thor** — a kernel
independently measured at 89% of spec reports ~30%. Bandwidth utilisation stays analytical
(byte model / elapsed).

## 4. Hard constraints from the operator

- **No cloud spend.** Everything runs on hardware already owned.
- **Delete nothing without explicit approval.** Prior-art repos live on this host.
- **Escalate before:** destructive disk operations, going past a 50% prune ratio, dropping vision
  capability, anything requiring cloud spend.
- When research contradicts the directive, **trust the research and flag the contradiction**.
- Prefer the validated path over the clever one. The priority is **preserved intelligence**, not
  maximum compression or speed.

## 5. Disk

98 GiB checkpoint, 129 GiB free at repo creation. Do not materialise a second full-precision copy
of the model. Requantisation work writes tensor-at-a-time and streams.

## §6 — measuring on a shared box

Two runs in this repo have now produced a number that looked like a finding and was contention:

- `bench_kda` reported "29.3% of achievable" against the idle-box 240 GB/s while an unattended job
  held the GPU at 96%. A streaming probe at that instant managed 82.8 GB/s: the kernel was at the
  ceiling (`OPTIMIZATION_LOG` #1).
- `bench_batch`, timing each width in one contiguous block, reported **K=2 as faster than K=1 in
  absolute ms/forward** — arithmetically impossible, and only visible because that comparison has a
  known sign.

So, for any timing on this box:

1. **Measure a streaming read in the same process, immediately before the timing loop**, and report
   everything against that number rather than against 240 GB/s.
2. **Visit the conditions round-robin, not in blocks**, so a contention spike lands on all of them.
3. **Report the minimum over reps, with the median beside it.** The true cost is a floor set by
   bandwidth; every disturbance can only push a sample above it. If min and median diverge, the run
   is not a measurement — say so and repeat it, rather than reading a trend into the noise.
4. **Include a comparison whose sign you already know.** The K=2 < K=1 result was caught only
   because monotonicity was predictable. A benchmark with no such anchor cannot tell you it is lying.

## §7 — do not pgrep or pkill a pattern that appears in your own command line

`pkill -f "[d]ense_nvfp4_probe"` killed its own shell. The bracket trick stops a pattern matching
*itself*, but the same command line also contained the literal path `tools/dense_nvfp4_probe.py`,
which the pattern matched. It happened a second time with `pkill -f 'build/glm5-server'` in a
command that also rebuilt `build/glm5-server`. Kill by PID, or put the kill in a command that names
nothing else.
