# PP Migration S2: Bounded Candidate Runtime Design

Status: implemented as an experimental local two-device prototype. The user confirmed the ownership boundary before implementation.

## Implementation checkpoint

The current worktree implements the bounded runtime ownership described below:

- `llama_context::placement_runtime` owns the placement, layer replicas, hybrid execution view, scheduler, reserve buffers, graph results, frontier, generation, and readiness flags.
- The context retains `active`, `candidate`, and one temporary rollback handle. Promotion swaps runtime handles and fixed state-bank bindings.
- Candidate scheduler reserve happens before publication. The commit path waits for state completion, validates generation and frontier, swaps bindings, and does not call `sched_reserve()`.
- Returning from A to B to A reuses the retained scheduler and fixed state-bank addresses.
- Candidate state synchronization uses backend asynchronous tensor copy plus an event. Event creation failure falls back to backend synchronization before the state can be published.
- Candidate CUDA Graph warmup executes the real candidate TG runtime on final tensor addresses. It snapshots and restores state data and metadata before returning.
- Shadow compute uses the complete candidate runtime, advances its private metadata view, preserves moved-layer bank updates, and restores state banks shared with the active runtime.

The implementation is intentionally narrower than the complete research plan. It supports the existing single-sequence, append-only, host-sampled Qwen hybrid placement path on local adjacent backends. It does not implement an activation journal, partial-layer replay, a tool-window admission controller, RPC transport, or non-blocking completion polling.

## Verification record

The implementation was built and tested on two Tesla T4 devices:

```sh
cmake --build build-pp-cuda --target test-llama-archs -j 8
CUDA_VISIBLE_DEVICES=0,1 build-pp-cuda/bin/test-llama-archs --pp-replicas
CUDA_VISIBLE_DEVICES=0,1 build-pp-cuda/bin/test-llama-archs --pp-model Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf --pp-low-vram
```

The fixture verifies exact state restoration after warmup, exact logits after async promotion, mixed recurrent/attention shadow tracking, stale request rejection, cancellation, append-only invalidation, rollback, and continued inference. CUDA diagnostics report warmup completion on the actual candidate graph.

The real Qwen3.5-35B-A3B run reported maximum absolute and relative logits differences of `0/0` for recurrent, full-attention, and mixed intervals. Its final commit-path scheduler reserve time was `0.000 ms`; handle and bank switching was `0.001-0.002 ms`. This does not prove copy/compute overlap or production latency isolation. Those require a timeline benchmark and the later admission controller.

## Outcome

S2 removes graph build and scheduler reserve from the migration commit path. Later work in the same prototype added asynchronous state copy, correctness-first shadow compute, and candidate graph warmup on top of this runtime object.

The context owns at most two TG runtimes:

- `active`: serves foreground decode.
- `candidate`: represents one validated adjacent placement and can be prepared without changing the active runtime.

The cache is deliberately bounded. Preparing a different candidate retires the previous candidate after its backend work completes.

## Runtime contents

Each runtime owns:

- The complete layer placement and immutable layer-replica pointers.
- One coherent hybrid execution view and its exclusive frontier.
- A per-layer binding to fixed KV or recurrent state banks.
- Its own backend scheduler and compute buffers.
- Its own graph-result storage for reserve and decode graphs.
- A shape signature for the supported TG decode form.
- Readiness flags kept separately for state, scheduler reserve, native CUDA Graph warmup, and frontier validity.
- A generation used to reject stale completion and promotion messages.

Weights remain model-owned. State banks remain memory-owned. A runtime borrows both through stable handles.

## Required ownership changes

The current context stores one global `sched`, `gf_res_prev`, `gf_res_reserve`, `active_layers`, and `layer_devices`. S2 groups these placement-dependent members under a private runtime object. Non-placement configuration and user-visible outputs remain context-owned.

Graph construction receives the runtime explicitly. It must resolve layer replicas and state-bank tensors from that runtime rather than temporarily rewriting global `active_layers` or the memory object's active tensor arrays.

The scheduler backend list can stay common when both runtimes use the same local devices, but scheduler instances and their allocated compute buffers cannot be shared. This is necessary because reserve, graph split state, and CUDA Graph keys belong to one placement topology.

## Lifecycle

### Create

1. Build the active runtime from the initial placement.
2. Reserve its supported TG shapes as today.
3. Preallocate fixed state banks for validated adjacent placements.
4. Create no candidate scheduler until a candidate is requested, unless eager preparation is enabled for measurement.

### Prepare candidate

1. Synchronize at frontier `P` in S2.
2. Clone the complete active metadata view at `P`.
3. Bind the target fixed banks and copy affected layer state into them.
4. Construct and reserve a candidate scheduler using the target placement without publishing it.
5. Record `state_ready(P)` and `runtime_reserved(signature)` separately.

The implemented prepare path enqueues step 3 asynchronously and reserves the candidate scheduler while the copy is in flight. The call itself returns only after reserve completes. Commit or cancel waits for the destination-ready event. Shadow compute can then advance the candidate frontier while active generation continues.

### Promote

Promotion requires:

- candidate generation is current;
- active and candidate frontiers are equal;
- candidate state banks are ready at that frontier;
- scheduler reserve matches the next decode signature;
- no foreground graph is in flight.

Promotion swaps the active and candidate runtime handles. It does not allocate tensors, copy state, build a graph, or reserve a scheduler. The old active runtime remains available for immediate rollback until the first promoted decode succeeds.

### Roll back and retire

If the first promoted decode or validation fails, swap the runtime handles back. A retired runtime releases its scheduler only after synchronization. Fixed state banks remain allocated for reuse.

## CUDA Graph boundary

Scheduler reserve and native CUDA Graph readiness remain separate. `warmup_layer_migration_candidate()` runs the candidate's real TG graph repeatedly, then restores all candidate-bound state and metadata in place. Shadow execution also marks the candidate graph ready only after the real full candidate runtime executes; there is no separate shadow-only graph.

## Minimum interface

The implementation should introduce private concepts equivalent to:

```text
placement_runtime
  placement
  layers
  memory_view
  bank_bindings
  scheduler
  graph_storage
  signature
  generation
  frontier
  state_ready
  reserve_ready
  graph_ready
```

The public synchronous migration method remains the reference API. During S2 it becomes `prepare candidate runtime -> promote -> validate`, so existing callers and exactness tests remain valid.

## Validation gates

- Candidate reserve does not mutate active placement, metadata, banks, scheduler, outputs, or logits.
- Promotion performs no backend allocation and no scheduler reserve.
- Repeated A/B/A promotion reuses both fixed state addresses and scheduler compute-buffer addresses.
- The first decode after promotion is exact and does not call reserve.
- Injected prepare, promote, and first-decode failures restore the previous runtime.
- A stale frontier or generation is rejected.
- Full-attention, recurrent, and mixed intervals pass on two devices.

## Confirmed ownership boundary

The implemented boundary is:

> Use two bounded full-placement TG runtimes, each with its own scheduler and complete hybrid view, while state storage remains in shared fixed per-layer/per-device banks. Promote by swapping runtime handles, not by mutating one global scheduler in place.
