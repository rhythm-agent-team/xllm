---
name: xllm-npu-profiler
description: Capture and export xLLM Ascend NPU traces, or analyze existing traces in Perfetto.
---

<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

# Generate an xLLM NPU Profile

Launch or reuse an xLLM server, validate a representative request, capture a short
NPU trace, and return the exported timeline with Perfetto analysis. If the user
already has a trace, skip server startup and capture and begin with Step 5.

## Prerequisites

- A working xLLM NPU build, model files, and available Ascend devices.
- CANN `msprof` in the service environment and access to the service PID namespace.
- A server launch command and a request workload appropriate for the model.
- For UI analysis, a browser on any machine that can access the exported trace.
  Capture, export, and command-line analysis do not require a browser or desktop.

## Choose the execution environment

Follow `AGENTS.md` and identify where the agent, xLLM service, and browser run.
The **execution environment** is the host or container running xLLM; the
**viewing machine** is where the user opens Perfetto. They may be the same machine.

| Agent location | How to execute the workflow |
|---|---|
| On the NPU server, with xLLM running on the host | Run capture and export commands directly in the server shell. No SSH or container is required. |
| Already inside the xLLM container | Run commands there directly; do not SSH back to the host or enter another container. |
| On the server host, with xLLM in a container | Enter the service container to use its tools and PID namespace. |
| On another machine | Connect to the NPU server through SSH, then enter a container only if the deployment uses one. |

Resolve checkout paths and script locations from the deployment. Record source and
native artifact identities as described in [capture setup](references/capture.md#1-confirm-the-execution-environment).
Reuse existing native artifacts when their native inputs, build configuration and
runtime environment still match. Python-only model edits that do not change native
inputs need a service restart, not recompilation; rebuild affected artifacts when
native inputs change. A prebuilt binary does not validate a new build.

A headless server can complete capture, export, and optional Trace Processor SQL
analysis. Return server artifact paths and transfer instructions for later UI
analysis; lack of a browser must not block device profiling or imply UI success.

## Step-by-step workflow

### Step 1: Launch or reuse the server

Check device availability with `npu-smi info` and inspect existing services before
launching. Use the task's model, device allocation, parallelism, and graph settings.
Create a unique run directory and save launch arguments, environment versions,
commit, executable path, and server PID in `manifest.md`.

Set this variable in the **server's launch environment**:

```bash
export PROFILING_MODE=dynamic
# Run the deployment's xLLM launch command in this environment.
```

For an existing server, confirm that the variable was present at startup. Setting
it only in the profiler or request client has no effect on that server. Arrange
any required restart within the task's authorization. Resolve the service parent
PID in the execution environment; a worker PID from `npu-smi` is not a substitute.

See [capture setup](references/capture.md#1-confirm-the-execution-environment)
for environment checks and launch pitfalls.

### Step 2: Wait for readiness and warm up

Poll the deployment's readiness endpoint with a bounded timeout, inspect startup
logs, and send warmup requests. Stop on startup or request errors. An open port
alone does not prove that model requests work. Complete graph capture and
compilation warmup before starting profiling.

### Step 3: Validate the workload

Verify that the requests match the intended workload. Record actual input and
output token counts, concurrency, prefix-cache settings, and early EOS behavior
in `workload.log`. If accuracy validation is requested, run the relevant accuracy
check before profiling; request success alone does not establish model accuracy.

### Step 4: Capture and export the profile

Follow [capture.md](references/capture.md#2-warm-up-then-capture-a-bounded-window)
for executable commands. The required order is:

1. Attach `msprof --dynamic=on` to the verified service parent PID in the same
   PID namespace, using a fresh output directory.
2. Wait for attachment readiness and enter `start`.
3. Run the profiling workload and verify all requests completed successfully.
4. Enter `stop`, confirm capture stopped, then enter `quit`. Wait for collector
   exit and data flush before exporting.
5. Export **every** `PROF_*` directory with `msprof --export=on` and retain logs.

For other profiling methods, check the current implementations of
`WorkerImpl::start_profile` and `WorkerImpl::stop_profile` in
`xllm/core/runtime/worker_impl.cpp`.

### Step 5: Inspect and optionally transfer the timeline

Locate complete `msprof_*.json` files, commonly under
`PROF_*/mindstudio_profiler_output/`. Existing `trace_view.json` or
`*.pt.trace.json` files are also candidates if their events are valid. Check for
nonempty timestamped events and record rank and device, size, and SHA-256. Keep the
trace in place if it is already accessible to the viewing machine; otherwise
transfer it and verify the destination hash. Preserve directories for different ranks.

Use [Perfetto analysis](references/perfetto.md) for command-line or UI inspection.
When UI inspection is requested and available, open the trace in Perfetto and
confirm nonempty tracks and selectable slices. Record the events, tracks, time
windows, units, and screenshots supporting each conclusion. If UI inspection
was not performed, state that explicitly; command-line parsing does not prove
browser visualization.

### Step 6: Clean up and report

Stop collectors and temporary trace processors created for this run. Stop only
a server started for this task when it is no longer needed; preserve reused
services and unrelated jobs. Record the service's final state.

Return timeline and report paths, their host or container, rank and device coverage,
capture and export status, and whether UI inspection was performed. Keep raw
`PROF_*` data and logs, including failure evidence. A useful artifact layout is:

```text
<run_id>/
  manifest.md           # Configuration, versions, executable, commit, parent PID
  capture.log           # Collector commands, timestamps, errors, exit status
  workload.log          # Warmup and profiling request results
  export.log
  PROF_*/               # Raw data; may remain remote with paths in the manifest
  timelines/            # Selected exports or verified copies organized by rank and device
  timeline_notes.md     # Observations, interval evidence, hypotheses, next steps
  screenshots/          # Overview and selected intervals when UI analysis succeeds
```

## Customization

- **Prefill:** use long inputs and short outputs.
- **Decode:** capture enough steady decode steps and exclude the initial prefill
  interval from decode measurements.
- **Multiple ranks:** export all ranks; start with one representative rank and
  compare matching steps and clocks when investigating skew.
- **Graph execution:** preserve the deployment's graph mode. Use an additional
  eager capture only when needed for operator mapping.
- **Longer captures:** extend only enough to cover the behavior of interest;
  check storage and trace size before increasing the window.

Profiling explains bottlenecks. Validate any claimed speedup with matched
measurements taken with profiling disabled.
