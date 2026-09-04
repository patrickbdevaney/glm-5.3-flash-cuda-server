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
