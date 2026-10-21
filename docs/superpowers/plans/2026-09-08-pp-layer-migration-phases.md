# PP Layer Migration: Phased Implementation Plan

**Status:** Planning complete. All implementation phases below are pending.

**Baseline:** `0ef4d560e`.

**Goal:** Move a contiguous group of Qwen3-Next/MoE layers between adjacent PP devices during generation, including devices on different RPC servers.

**Architecture:** Preload candidate weight replicas, track execution placement per context, mirror mutable layer state in the background, prepare a separate execution graph, and switch at a compatible decode boundary without waiting for migration readiness. Retain weight replicas and retire obsolete state and execution resources after their final use.

**Tech stack:** C++, GGML backend scheduler, CUDA, GGML RPC, existing llama.cpp tests and examples.

**Spec:** This document preserves the decisions from the accepted PP migration plan and defines the implementation order and acceptance gates.

## Fixed decisions

- Support one active context per participating model and one sequence per context. Start with Qwen3-Next and local CUDA / RPC-backed CUDA devices.
- Move complete layers, including all routed experts, shared experts, routing tensors, normalization tensors, and weight scales. Keep embedding and output placement fixed.
- Restrict moves to a contiguous interval at an adjacent PP stage boundary. Runtime discovery of new devices is outside this version.
- Preload each candidate layer on every configured destination before generation. Selecting all layers as candidates permits full weight redundancy where memory permits.
- Retain source weight replicas after a move. Reclaim source state, snapshots, and unused execution resources; do not claim that layer weight memory was released.
- Interpret a requested position as the earliest permissible switch. State version `t` contains the input at sequence position `t` and can be used to process position `t + 1`.
- Do not wait on migration readiness in the generation thread. Preparation can affect shared compute and network resources; unchanged token latency and eventual readiness under insufficient bandwidth are not guaranteed.
- Disable MTP, speculative decoding, rollback, context shift, runtime adapter changes, and backend sampling in this experimental path. Use ordinary append-only generation and host sampling at the existing output synchronization boundary.
- Expose explicit requests, status, and cancellation. Automatic load balancing is outside this version.
- Feature behavior is opt-in. Preserve the current path when disabled.
- Extend existing tests. Do not create new files in `tests/`, commit, push, or submit changes as part of this plan.

## Dependency order

| Phase | Deliverable | Depends on |
| --- | --- | --- |
| 1 | Reproducible baseline and diagnostics | None |
| 2 | Weight replicas and context placement | 1 |
| 3 | Independent layer state and a synchronous test reference | 2 |
| 4 | Coexisting active and candidate execution graphs | 3 |
| 5 | Background state replication with version tracking | 3, 4 |
| 6 | Nonblocking boundary switch and deferred retirement | 4, 5 |
| 7 | Public integration and complete local/RPC acceptance | 6 |

Implement in this order. Complete the acceptance gate for each phase before using it as the basis of the next phase. The synchronous reference in phase 3 is a test oracle, not the final user-facing migration behavior.

## Phase 1: Baseline and diagnostics

**Outcome:** A repeatable comparison that can distinguish model correctness, placement errors, and migration overhead.

**Main locations:** Existing `tests/test-save-load-state.cpp` and `tests/test-rpc-multi-server.cpp` cases, plus existing context/scheduler debug logging.

- [ ] Add an opt-in single-sequence baseline case using a fixed token input stream. Record logits, sequence positions, model/precision settings, initial layer ownership, and KV/R/S allocation sizes.
- [ ] Exercise a normal local multi-device split and a split across two RPC endpoints. Preserve the existing multi-server negative pointer test.
- [ ] Record normal token intervals and graph reuse behavior without a migration request. Keep device timing and memory diagnostics behind the experimental/debug path.
- [ ] Save the exact baseline invocation, GGUF identity, device names, build flags, and test result for later comparisons.

**Acceptance:** The selected model runs correctly on both baseline layouts. Required hardware/model tests are explicitly marked skipped if unavailable; a skip does not count as successful hardware acceptance.

## Phase 2: Weight replicas and context placement

**Outcome:** The same logical layer can use a different preloaded physical weight replica when a context is created, without an in-flight migration yet.

**Main locations:** `llama_model` and its loading helpers, `llm_graph_context`, Qwen3-Next graph construction, and context device selection.

- [ ] Record logical tensor identity and layer ownership while loading. Create replicas for configured destinations using explicit tensor mappings, including scale tensors, aliases, and views.
- [ ] Add context-owned layer placement and an epoch. Initialize it from the existing split unless an experimental initial placement is provided.
- [ ] Route Qwen3-Next layer weight accesses through an active-layer accessor. Audit helper accesses, layer callbacks, state placement, and fused-op probes for reads of the original model placement.
- [ ] Validate candidate devices and capacity before generation. Select an operator configuration supported by all allowed destinations; reject placements that require unplanned CPU fallback for migrated layer computation.
- [ ] Compare contexts created with two different initial placements. Verify complete expert coverage and that recurrent decode does not copy model weights between devices.

**Acceptance:** Static placement changes produce equivalent results, every layer uses the selected replica, and disabling the feature preserves existing behavior. No runtime pointer rebinding of an already allocated tensor is used.

## Phase 3: Independent state and synchronous migration reference

**Outcome:** A frozen context's state can be relocated correctly, with a known-good result to compare against the later asynchronous implementation.

**Main locations:** `llama_kv_cache`, `llama_memory_recurrent`, `llama_memory_hybrid`, and the existing state test harness.

- [ ] In the experimental path, allocate each movable layer's state independently. Preserve physical layout and rebuild K/V stream views against destination tensors.
- [ ] Implement destination state allocation and copying. Include the required K/V history, all live R/S state, and consistent CPU cell/head/position metadata.
- [ ] Add an internal test-only operation that synchronizes at a decode boundary, copies state, selects destination weights/state, rebuilds execution resources, and resumes the same token stream.
- [ ] Verify linear-only, full-attention-only, mixed-interval, and reverse moves locally and across RPC endpoints.
- [ ] Reclaim the moved source state's independent buffers after the test reference has released its old graphs and copies. Keep source weights resident.

**Acceptance:** Copied state is byte-identical before destination computation, continued inference agrees with the baseline, and source state allocations can be reclaimed without freeing unrelated layers. The blocking reference is not exposed as the final nonblocking API.

## Phase 4: Active and candidate execution graphs

**Outcome:** A new layout can be prepared while the old layout continues to execute.

**Main locations:** `llama_context`, `llm_graph_params` / `llm_graph_result`, backend execution resource lifetimes, and the RPC graph protocol.

- [ ] Group graph metadata, scheduler, execution handles, and retained state references into separate active and candidate resources. Share immutable weight data, not mutable graph metadata or allocation records.
- [ ] Build candidate graphs from independent metadata snapshots. Add a non-mutating preview of the next memory layout; candidate preparation must not advance the active sequence or write its state.
- [ ] Extend reuse checks with placement epoch while retaining current batch, attention-mask, and recurrent-head checks. Reject a candidate that does not match the actual next decode.
- [ ] Extend RPC with prepare-without-execution, execute-by-handle, and release operations. Keep the active and candidate split graphs addressable simultaneously, with capability/version checks before enabling the feature.
- [ ] Prepare graph allocation and remote registration in the background. Do not warm up against live state; native backend first-use costs remain part of measured inference latency.
- [ ] Ensure retiring graph handles and native backend caches cannot leave executable references to freed state or temporary buffers.

**Acceptance:** Preparing, rejecting, or destroying a candidate leaves active inference numerically unchanged. Remote candidate registration does not overwrite the active graph. Graph preparation performs no model-state computation.

## Phase 5: Background state replication

**Outcome:** The destination follows the source's evolving state while generation continues, without taking ownership yet.

**Main locations:** Layer-state snapshot helpers, the scheduler's internal split completion path, and RPC transfer submission/completion.

- [ ] Copy the committed K/V prefix first, then transfer newly committed rows. Preserve row offsets and publish a version only when its entire required history is present.
- [ ] Capture complete R/S snapshots after the candidate layer's state writeback. Use two preallocated source snapshot slots and completion events; never transfer from mutable state while a later token can overwrite it.
- [ ] Add an internal nonblocking completion hook near the candidate region's end. Do not use the existing synchronizing evaluation callback. Enqueue source snapshot copies in the correct compute-stream order.
- [ ] Implement asynchronous submission for remote snapshot copies. Perform cross-server transfers through the coordinator with a bounded 4 MiB staging block per request and at most one such transfer in flight per endpoint for this request.
- [ ] Let the worker wait for transfer completion and publish readiness. The generation thread only enqueues available work and reads status; if no snapshot slot is free, skip that snapshot and continue.
- [ ] Track readiness for the complete migrated interval. A newer complete R/S snapshot may replace an older one; missing K/V rows must be transferred before the newer interval version is ready.
- [ ] Protect destination state updates with explicit request ownership. A ready version cannot be overwritten after the generation thread has claimed it for a switch.

**Acceptance:** Under delayed transfers and snapshot-slot pressure, source generation continues correctly, snapshots are not overwritten, and each published ready version matches the reference state. On a configuration with enough overlap, at least one candidate reaches the current token boundary.

## Phase 6: Nonblocking switch and deferred retirement

**Outcome:** Execution actually moves between devices without a new migration wait in the generation loop.

**Main locations:** Context decode boundary selection, placement publication, request ownership, and local/RPC resource retirement.

- [ ] Check eligibility at the existing boundary after output synchronization and before applying the next decode's memory update. Require complete state for the current position, a compatible candidate graph, completed destination writes, and no unfinished old-graph state access.
- [ ] Atomically claim a ready request. If state or graph readiness is stale or busy, continue with the old layout; do not synchronize, rebuild, allocate, or wait for migration at this decision point.
- [ ] Publish the new placement, state bindings, and execution resources together, then advance the placement epoch. Prevent queued migration writes from modifying the new active state.
- [ ] Retire old resources on the worker after their last events and references complete. Release remote graph handles before referenced buffers. Retain source weight replicas.
- [ ] Exercise the same mechanism locally first and then with two distinct RPC servers. Test reverse migration by creating a fresh state mirror on the original device.

**Acceptance:** Local and cross-server moves complete without token loss, duplication, or use of stale state. Late candidates delay migration rather than token generation. Old state and unused execution allocations are retired safely, and reverse moves do not reload weights.

## Phase 7: Integration and final acceptance

**Outcome:** An opt-in interface that another scheduler or application can use and a complete evidence set for its supported behavior.

**Main locations:** The public model/context API and its existing implementation files, an existing example, and existing backend/state/RPC tests.

- [ ] Expose pre-generation replica configuration and runtime request/status/cancel operations. A request specifies layer interval, source, destination, and earliest sequence position; only one request is outstanding per context.
- [ ] Report request stage, destination state position, placement epoch, actual switch position, pending retirement, and recoverable preparation errors. Cancellation before commit leaves the active layout unchanged.
- [ ] Integrate an explicit migration trigger in an existing example. Do not add an automatic balancing policy or server endpoint in this phase.
- [ ] Run fixed-input inference comparisons using the same CUDA backend family and precision, with `atol=1e-3` and `rtol=1e-3` for logits. Check migrated valid state bytes before computation; do not substitute generated-text equality for tensor checks.
- [ ] Cover linear/full/mixed moves, multiple expert routes, K/V mask size changes, recurrent-head changes, late candidates, cancellation, repeated forward/reverse moves, insufficient destination memory, and missing replica/backend/RPC capabilities.
- [ ] Confirm the disabled path still passes its existing tests. Record local and real cross-machine token intervals, snapshot/transfer bytes, version lag, preparation time, and resource retirement.
- [ ] Document that fatal RPC disconnect recovery, guaranteed switch deadlines, unchanged token latency, and automatic throughput improvement are outside the acceptance promise.

**Acceptance:** Both local CUDA and real cross-machine RPC runs complete the supported scenarios. No migration-specific readiness wait occurs in the generation thread. Source state is released after retirement; retained weights are reported separately. Missing hardware or model coverage remains an explicit outstanding acceptance item.

## Milestones and handoff

- **After phase 3:** Functional correctness reference. Weight selection, state relocation, graph rebuilding, and state reclamation have been tested together, but migration still pauses in the internal reference.
- **After phase 5:** Background preparation is independently demonstrated. Active generation continues while the destination catches up, but execution ownership has not yet moved.
- **After phase 6:** The complete migration mechanism works locally and across RPC.
- **After phase 7:** The experimental interface, diagnostics, and acceptance record are ready for application use.

At each handoff, record the actual changes, tests run, unavailable coverage, remaining limitations, and the next phase. Do not mark a phase complete solely because its code compiles, and do not report planning or a skipped hardware check as an implemented migration feature.
