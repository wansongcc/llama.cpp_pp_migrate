# PP State Reconstruction and Migration Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans when implementation is requested. Steps use checkbox syntax for tracking. Repository instructions override skill defaults: do not commit, push, create PRs, or add new test source files.

**Goal:** Reduce the total communication cost of moving hybrid-attention layers between adjacent PP devices while keeping generation within its latency and task-quality budgets.

**Architecture:** Preload a bounded set of candidate layer weights. Maintain a consistent destination checkpoint and a bounded log of the migrated interval's input activations. Catch up through either state copying or suffix replay, using tool idle periods and measured device slack. Commit only at a current, fully prepared decode boundary. Add linear-state composition as a measured optional optimization.

**Tech Stack:** C++, GGML graphs and backend schedulers, local CUDA first, GGML RPC second, Qwen3-Next and Qwen3.5 MoE.

**Spec:** The Design Contract in this document is the proposed specification. This is a development proposal, not an implementation or hardware acceptance report.

**Baseline:** `codex/pp-phase123`, commit `3cfc792db86d55311df15b93d69ccba6176aec8c`, inspected again on 2026-10-17.

## Global Constraints

- One active context per participating model, one append-only sequence, host sampling, layer split, adjacent devices, contiguous boundary intervals, fixed embedding/output placement.
- Disable speculative decoding, MTP, context shifting, sequence editing, runtime adapter changes, and backend sampling for the first version. Reverse device migration is supported; token-history rollback is a different feature.
- Keep current tensor precision in the correctness version. Hybrid recurrent R/S are currently FP32, and the existing layer-input extraction uses host float buffers. BF16/FP8 transport is a separate experiment.
- Reuse existing weight replicas, independent state allocation, placement checks, graph helpers, and test files. Do not instantiate a second public llama_context on the exclusive model to implement replay.
- Keep the disabled execution path unchanged. Bound candidate count, log memory, snapshot slots, transfer blocks, and background work per scheduling opportunity.
- Preparation failures leave the source serving. A lagging replica delays migration rather than forcing a readiness wait into generation. Insufficient resources can prevent migration indefinitely; report this explicitly.
- Preserve the existing `2026-09-08-pp-layer-migration-phases.md`. Its all-pending status predates code now present; neither its checkboxes nor commit descriptions constitute test evidence.

## Review Focus

- Live recurrent state overwritten during an asynchronous snapshot: require an immutable snapshot and completion ordering; tested in Task 2.
- Activation ring gaps, eviction, stale epochs, and half-open position mistakes: refuse replay and leave active state unchanged; tested in Task 3.
- Replay sharing active cells, graph metadata, or allocation buffers: source sequence/state must remain unchanged; tested in Task 4.
- Stale candidates, changed mask dimensions, and outstanding writes at cutover: refuse commit without a generation-thread wait; tested in Task 5.
- RPC graph cache replacement and FIFO head-of-line blocking: active and replay graphs must remain distinct and latency must be measured; tested in Task 7.

## Current Code Evidence

| Capability | Evidence | Remaining gap |
| --- | --- | --- |
| Candidate whole-layer weights | `src/llama-model.cpp:1872`, `prepare_layer_replica()` | Must be prepared before context creation; each MoE replica includes expert weights |
| Per-context placement and epoch | `src/llama-context.cpp:104`, `src/llama-graph.h:814` | Experimental single-sequence QWEN3NEXT/QWEN35MOE restrictions |
| Candidate validation | `src/llama-context.cpp:594` | Enumerates initial-layout single-boundary alternatives, not every reachable multi-boundary layout |
| Synchronous migration reference | `src/llama-context.cpp:698` | Synchronizes, allocates/copies state, resets graphs, swaps placement, reserves again |
| Per-layer state storage/copy | `src/llama-kv-cache.cpp:1334`, `src/llama-memory-recurrent.cpp:217` | Copies complete tensors; no live-row delta protocol, immutable snapshot queue, or private replay metadata |
| Hidden-input capture | `src/models/qwen35moe.cpp:253`, `src/models/qwen3next.cpp:226`, `src/llama-context.cpp:2500` | Current-batch host extraction, not a persistent replay log or an arbitrary-layer input API |
| Model forward builders | `src/models/qwen35moe.cpp:230`, `src/models/qwen3next.cpp:211` | Build the normal model pass; no independently scheduled arbitrary interval replay |
| Existing migration tests | `tests/test-llama-archs.cpp:894`, `:1239`; `tests/test-save-load-state.cpp:1114` | Existing synchronous/replica/baseline tests do not demonstrate async replay or bubble overlap |
| Async copy fallback | `ggml/src/ggml-backend.cpp:511` | Can synchronize both backends and execute a blocking copy |
| RPC execution | `ggml/src/ggml-rpc/ggml-rpc.cpp:557`, `:585`, `:1749` | Per-endpoint dispatcher FIFO; synchronous server compute; one stored recompute graph per device |

The inspected `build-pp/CMakeCache.txt` has CUDA ON, RPC OFF, and tests ON. `ctest --test-dir build-pp -N -R 'llama-archs|save-load-state|rpc'` listed zero tests. Use explicit binary invocations after checking/building targets; this inspection did not execute inference tests.

## Design Contract

### Position and ownership

Use an exclusive frontier `F`: all inputs at positions `[0, F)` have completed, and position `F` is next. A checkpoint at `P` contains state after position `P-1`. Replay consumes `[P, F)`. The older plan's inclusive state version `t` maps to `F=t+1`.

Each request identifies a sequence, immutable model/config identity, source placement epoch, interval `[first, end)`, destination device, and earliest permitted frontier. Permit only one outstanding request per context initially. Predeclare a small finite set of neighboring layouts; reject plans outside it. Supporting successive different-boundary changes requires extending the current candidate catalogue, not just having weight replicas.

Three resources are separate:

1. Active execution: live tensors, sequence cells, graph results, scheduler, output and sampler state.
2. Candidate state: destination tensors, private cell/head/position metadata, checkpoint frontier and caught-up frontier.
3. Candidate execution: private replay graph/scheduler/scratch and prepared normal-decode resources for the new layout.

Only immutable weights are shared freely. Unmoved live state may be referenced by a prepared decode plan, but preparation must not execute writes to it. All mutable ownership changes occur on the generation thread at an existing safe decode boundary.

### Checkpoint and input requirements

For the migrated interval, a checkpoint includes the required full-attention KV prefix, linear R/S state (including causal convolution history), any other actually present state tensors, and consistent metadata. Stage replay starts with every participating layer at the same checkpoint frontier. A newer recurrent endpoint cannot simply replace an older one while replay still needs intermediate outputs from that older point.

Capture the pre-normalization residual input of the first migrated layer, not just the old PP stage boundary. Log every replayed position with sequence, position, layer, dtype/layout, model/config identity, epoch, and completion information. Own the storage; never retain only a graph tensor pointer. Require each chunk to be locally readable by the executing backend before consuming it. The whole suffix need not reside there at once: transfer and consume it in ordered chunks.

For downstream-to-upstream moves, the destination may already produce the required activation; retention is still necessary. For upstream-to-downstream moves, its usual input is often the moved interval's output, so capture/transfer the earlier activation explicitly. A device that once produced a value does not necessarily still hold it, and a logical layer device does not prove the actual tensor resides there.

KV history in this append-only version can be transferred in completed row spans. Handle transposed V layouts, quantized block boundaries, padding and cell-to-position mapping. The destination may allocate full capacity while transferring only initialized/live ranges; initialize unused storage as required. Do not send the full prefix at every checkpoint refresh: extend KV coverage and refresh recurrent snapshots.

Snapshot recurrent tensors into a bounded source slot after their writeback and before the next mutation. Copy out of that immutable slot. Publish a checkpoint only after all moved layers and metadata represent the same frontier. Skip/defer a refresh when no slot is free.

### Two complete catch-up modes

- `COPY`: extend destination KV to a chosen frontier and install the matching latest recurrent snapshot. No activation replay is required.
- `REPLAY`: restore the checkpoint and run the actual migrated interval on its logged input suffix. Intermediate layers run attention, residual paths and FFN/MoE so later layers receive valid inputs. It reconstructs both KV and recurrent state together.

Choose a complete dependency-valid plan. Do not independently choose final-state composition for a middle linear layer and KV replay for a later layer while omitting the intermediate token outputs. A later optimization may end replay at a final layer's state-producing nodes when its output is unnecessary. For a single full-attention layer with known inputs, generating its KV needs the projection/norm/position path, not prefix attention; full interval replay requires prefix attention to produce subsequent layers' inputs.

Replay never samples, advances the active sequence, emits user output, or repeats tool calls. Initially retain normal per-layer computation for reference correctness; optimize dead outputs afterward.

### Readiness and cutover

`UNPREPARED -> CHECKPOINTING -> CATCHING_UP -> READY -> COMMITTED -> RETIRED`

Preparation can also become CANCELLED or FAILED. READY is a versioned condition, not a permanent assertion: source progress can make it stale.

At the existing output synchronization boundary, before applying the next decode memory update, commit only if the candidate frontier equals the live frontier, all required writes are complete, the new decode graph matches the next batch/mask/memory layout, the source epoch still matches, and no old execution still uses resources being replaced. Claim/seal the candidate first so a queued background operation cannot overwrite its now-active state. Publish placement, state bindings and execution resources together, then advance the epoch.

An earliest requested frontier is not a historical switch point. If source reaches 120 while candidate is at 110, it must catch up to a later current boundary; it cannot take over position 121 using state 110. Do not freeze a past cut position while allowing source to advance indefinitely and then call the candidate ready.

No migration payload transfer, allocator reserve, graph rebuild or wait-for-readiness belongs in this final decision. Metadata checks, prepared pointer/bundle publication and existing decode synchronization still have a cost. Measure the first post-switch token too, including native graph/kernel first-use work. Warmup may execute only against disposable private state.

Retain old resources until readers and events finish. Existing failure rollback before destination computation can remain supported. Keeping stale source state after successful new-token execution does not provide instant rollback: catching the source up would require another migration/replay protocol. Fatal device/RPC failover is outside the first version.

### Scheduling and cost

Tool idle is an application-provided opportunity, not an automatic property inferred from text. The caller explicitly signals idle entry and exit while keeping the sequence context alive. Tool completion has priority; long idle windows favor batching KV refresh and larger replay chunks. Short measured device gaps favor small bounded chunks. Low-priority streams do not guarantee kernel preemption, zero memory-bandwidth contention, or free computation.

For suffix length `D`, estimate outstanding costs using actual formats:

`B_copy = D * sum_FA(bytes_K_per_token + bytes_V_per_token) + sum_linear(bytes_latest_state)`

`B_replay = bytes_missing_boundary_activations + required_checkpoint_refresh_bytes`

`T_replay = measured_interval_time(prefix_length, D, batch_shape, backend, kernels)`

Do not model full-attention replay time as constant per token independent of prefix length. For activations already on the destination, network bytes can be zero but capture/storage/device-copy costs remain. Include checkpoint establishment, canceled candidates, source-to-coordinator and coordinator-to-destination traffic, and weight preload separately in accounting.

Select the lower-cost feasible mode subject to memory, observed background capacity, migration completion target, and foreground token-latency budget. Initial policy uses measured time/bytes and a simple threshold; do not start with a learned scheduler. Sustained effective catch-up throughput must exceed new work arrival rate to reduce existing lag. If it cannot, defer to tool idle, change mode, or report not-ready. Bound the journal: never silently overwrite needed input; invalidate/rebase or cancel the affected candidate instead.

### HYPIC scope and quality

HYPIC caches a segment transition and zero-start endpoint so that `S_F = A_[P,F) * S_P + U_[P,F)`. Its layer-exact claim assumes the same layer inputs; independent segment reuse can change deeper-layer inputs. Qwen convolution history also requires treatment. A combined endpoint alone does not emit every intermediate token output. Source: [HYPIC, sections 4.2 and 4.3](https://arxiv.org/html/2607.01299v2).

For this project, first evaluate same-request summaries derived from actual inputs as an optional endpoint reconstruction mode. Keep R/conv state and model-specific orientation/head mapping correct. Dense transition plus endpoint can exceed a direct state transfer; creating and applying the summary is part of its cost. Use it when a reusable summary already exists or measured amortization justifies it. Cross-prefix PIC/seam correction is a separate approximate experiment, not required for same-prefix migration.

Operational state/position correctness remains strict. The requested semantic criterion means no material task-quality regression under a declared evaluation and tolerance; it is not a mathematical guarantee of identical sampled outputs. Establish native-precision replay first. Compare teacher-forced logits/states and matched task trials. Do not adopt the earlier illustrative 90% top-k / 95% token agreement thresholds as a proof of correctness. Lossy transport needs a separate accuracy gate.

## Implementation Tasks

### Task 1: Establish the reference and measurements

**Files:** Existing `tests/test-llama-archs.cpp`, `tests/test-save-load-state.cpp`, `src/llama-context.cpp`, and this plan's acceptance record.

**Interfaces:** Preserve `migrate_layers_sync(int32_t first, int32_t end, ggml_backend_dev_t destination, bool fail_after_switch)` as the reference. Produce named metrics for state bytes, useful KV bytes, elapsed snapshot/copy/reserve time, baseline token intervals, and actual layout/precision.

- [ ] Build and run the existing local synthetic replica/migration test and the real Qwen35MoE path where hardware/model exist. Record model checksum, device identity, compile flags and skips.
- [ ] Extend the existing test harness with per-phase counters and checkpoint-frontier assertions; distinguish allocated tensor bytes from live transferred bytes.
- [ ] Compare no migration and synchronous migration on the identical accepted token stream. Preserve existing numerical comparisons; do not loosen thresholds just to accept replay.

**Gate:** A reproducible local oracle and measured migration bottleneck. This is not an async milestone.

### Task 2: Separate state allocation, snapshot and publication

**Files:** `src/llama-memory.h`, `src/llama-memory-hybrid.{h,cpp}`, `src/llama-kv-cache.{h,cpp}`, `src/llama-memory-recurrent.{h,cpp}`, existing state/architecture tests.

**Interfaces:** Introduce an internal owned `llama_layer_checkpoint` containing interval, sequence, exclusive frontier, source epoch, tensor storage and copied memory metadata. Add a non-mutating `prepare_layer_checkpoint(first, end, destination)` that allocates a private destination and a snapshot operation that returns completion status. Keep the synchronous wrapper implemented through these primitives.

- [ ] Add tests for completed KV spans, transposed V, R/S snapshots racing subsequent tokens, and copied cell/head metadata; deliberately delay transfer completion.
- [ ] Split current `prepare_layer_move()` allocation/copy/swap responsibilities; retain existing swap reference behavior.
- [ ] Add two bounded source recurrent-snapshot slots initially, correct writeback-to-snapshot ordering, and destination publication only after complete matching-frontier data.
- [ ] Compare checkpoint bytes with the frozen reference; assert canceled/failed preparation changes neither active placement nor active state. Reject unsupported layouts explicitly.

**Gate:** A private, consistent historical checkpoint can coexist with continued source execution.

### Task 3: Persist candidate interval inputs

**Files:** `src/llama-context.{h,cpp}`, `src/llama-graph.{h,cpp}`, Qwen builders only where capture placement requires it, existing tests.

**Interfaces:** Introduce internal `llama_layer_input_journal` and `llama_layer_input_span`. A span owns/retains data, layer id, sequence, `[begin,end)`, dtype/strides, epoch/config identity and completion state. `acquire(begin,end)` succeeds only for covered, ready positions; release permits eviction.

- [ ] Test a known interval against captured pre-norm inputs; include multiple microbatches, nonzero start position, source buffer reuse, and both migration directions.
- [ ] Initially copy existing FP32 extraction into a bounded persistent host journal after proper completion. This is the correctness path and may synchronize; label its measured overhead.
- [ ] Add device-resident/pinned staging and event-driven capture for the performance path. Capture within the producing backend's scheduled order so later compute cannot overwrite the input first.
- [ ] Test missing/duplicate/out-of-order spans, stale epochs, cancellation and capacity overflow. No gap may be concealed. A pinned needed span cannot be overwritten; rebase/cancel the candidate when bounded resources are exhausted.

**Gate:** Replay input lifetime and exact coverage are demonstrable; producing an activation once is no longer confused with retaining it.

### Task 4: Build isolated interval replay

**Files:** `src/llama-graph.{h,cpp}`, `src/models/models.h`, `src/models/qwen3next.cpp`, `src/models/qwen35moe.cpp`, `src/llama-context.{h,cpp}`, private memory helpers from Task 2, existing tests.

**Interfaces:** Define internal `llama_layer_replay` owning its graph, scheduler, scratch and candidate metadata. `replay(checkpoint, input_span)` advances only candidate state and returns a completion ticket; it publishes the new frontier after completion. Graph parameters include interval, input mode, candidate state identity and layout epoch.

- [ ] Add suffix-vs-uninterrupted-forward tests for linear-only, full-only and mixed intervals; compare all moved states and continued logits, and checksum unmoved/live state for noninterference.
- [ ] Factor reusable per-layer construction from the existing Qwen graph loops; accept pre-norm hidden input at `first`, stop at `end`, omit vocabulary projection and sampling, and retain every token output needed by the next layer.
- [ ] Use independent memory cells, batch metadata and scratch. Reuse immutable weight mappings without violating the model's exclusive context guard.
- [ ] First run single-token replay, then bounded chunk replay with causal masks and position ids. Cover convolution boundary history, recurrent heads and KV padding/mask-size transitions.
- [ ] Prove source inference is unchanged while candidate replay executes. Optimize the final layer to state-only work only after the full-interval reference passes.

**Gate:** Native-precision suffix replay reconstructs a candidate from a historical checkpoint without advancing source state. This still does not prove latency overlap.

### Task 5: Prepare the next layout and commit without readiness waits

**Files:** `src/llama-context.{h,cpp}`, `src/llama-graph.{h,cpp}`, existing tests; scheduler code only where lifecycle separation requires it.

**Interfaces:** Internal `llama_execution_bundle` owns per-layout scheduler/graph/allocation state. Define `llama_migration_request { first, end, destination, min_frontier, mode }` with COPY/REPLAY modes, `uint64_t request_layer_migration(const llama_migration_request &)`, `llama_migration_status get_layer_migration_status(uint64_t) const`, `void cancel_layer_migration(uint64_t)`, and `bool try_commit_layer_migration(uint64_t, llama_pos live_frontier)`.

- [ ] Add stale-frontier, outdated-epoch, next-mask-shape-change, not-complete-copy and canceled-request tests; every failed readiness check must keep generation on the source.
- [ ] Keep active and prepared decode bundles alive simultaneously. Prepare allocation and compatibility checks without swapping live memory or running warmup against it.
- [ ] Move request claim, current-frontier validation and bundle/state/placement publication to the existing safe decode boundary. Stop candidate writers before publication and increment placement epoch once.
- [ ] Retire old graphs/buffers only after last use. Test injected pre-commit failure and repeated forward/reverse moves within the predeclared layout set.
- [ ] Record cutover and first-token costs. If the current graph API still requires reserve/rebuild at switch, retain the functional path but mark the nonblocking gate unmet until prepared-bundle selection removes it.

**Gate:** A prepared current-version candidate can take over without payload copy, reserve, or readiness wait on the generation thread. Unready candidates cannot switch.

### Task 6: Schedule local CUDA preparation in actual slack

**Files:** `src/llama-context.{h,cpp}`, `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu` only for missing event/submission support, existing tests/harness.

**Interfaces:** Add a bounded `poll_layer_migration(work_budget_us)` service path and `set_migration_idle_hint(bool)` to the internal coordinator. Foreground execution owns active scheduler mutation; the background path owns candidate work. Status reports source/destination frontier, bytes, pending work, readiness and refusal reasons.

- [ ] Test copy and replay scheduling with injected zero slack and long kernels; source continues, candidate may remain pending, and capacity stays bounded.
- [ ] Implement incremental KV synchronization and chunked replay with explicit completion tickets. Detect synchronous backend fallbacks rather than calling generic async copy and assuming overlap.
- [ ] Measure effective catch-up rate, capture overhead and TTIT impact. Choose COPY or REPLAY using measured complete-plan cost and input availability.
- [ ] Integrate explicit tool-idle enter/exit in the harness. Prioritize resumed decode and cap background submissions; first use small chunks rather than relying on stream priority to preempt an already running kernel.
- [ ] Evaluate short-term shadow updates only for an imminent move, as an optional mode; account for their activations, compute and memory traffic.

**Gate:** Timelines show useful work in measured slack and lower migration bytes/stall on at least one declared workload without exceeding its foreground budget. Low slack can legitimately select COPY or defer migration.

### Task 7: Make the RPC execution model support the protocol

**Files:** `ggml/src/ggml-rpc/ggml-rpc.cpp`, RPC transport files only if needed, `ggml/include/ggml-rpc.h` if capability exposure changes, existing `tests/test-rpc-multi-server.cpp` and `.sh`, existing architecture/state tests.

**Interfaces:** Versioned optional RPC operations `GRAPH_PREPARE`, `GRAPH_EXECUTE_HANDLE`, and `GRAPH_RELEASE`, with server-owned session-scoped graph handles and explicit completion status. Preserve the legacy path for unsupported peers. Preparation must not execute the model.

- [ ] Add an interleaving test with active, replay and prepared-next-layout graphs. Each execute must select the correct graph, including after a different graph has run.
- [ ] Replace the single last-graph dependency for this feature with graph handles and explicit lifetimes. Verify handles and buffer ownership; opening another connection must not bypass remote-pointer validation.
- [ ] First support bounded serialized chunks at observed endpoint idle windows and record their blocking cost. This is a valid functional stage, not evidence of simultaneous same-endpoint copy/compute.
- [ ] If measurements require stronger overlap, add a separate capability for queued server-side async work, completion and controlled transfer/compute scheduling. Preserve session-owned buffers and foreground priority; a host worker alone is insufficient.
- [ ] Test two distinct servers, cross-server staging bytes, delayed completion, cancellation and graph retirement. Report fatal disconnect recovery as unsupported unless explicitly implemented later.

**Gate:** Correct graph isolation and real cross-machine measurements. Do not infer RPC overlap from advertised async/events flags. Local CUDA success does not satisfy this gate.

### Task 8: Evaluate linear-state composition, then expose the experiment

**Files:** Qwen builders and delta-net helpers identified by Task 4, `src/llama-memory-recurrent.{h,cpp}`, migration coordinator, `src/llama-ext.h` for an opt-in experimental interface after stabilization, existing tests and one existing example if useful.

**Interfaces:** A model-specific owned `llama_linear_segment_summary` identifies layer, sequence/input identity, position interval, dtype and conv boundary state, and stores the transition/write summary. Its compose operation advances only a compatible candidate endpoint; it cannot advertise intermediate hidden outputs.

- [ ] Compare direct state transfer, ordinary recurrence replay and summary construction/application on actual Qwen GDN dimensions. Count bytes, compute, storage and summary reuse, including construction on cache misses.
- [ ] Verify composition against recurrence with identical recorded layer inputs and conv history before testing cross-prefix approximation. Test segment order, missing summaries and precision drift.
- [ ] Enable summary composition only for a dependency-valid plan: an endpoint-only layer, already available downstream inputs, or already prepared downstream state. Do not bypass a middle layer needed to reconstruct downstream KV.
- [ ] Add opt-in request/status/cancel and application idle hints after internal tests stabilize. First replay recorded agent timing/tool traces in the harness; live tool orchestration remains application-owned.
- [ ] Evaluate all ablations below. Gate lossy hidden/KV transport and cross-prefix PIC behind separate task-quality evidence.

**Gate:** Keep the composition mode only where it provides measured net benefit. A negative result on one-shot dense-state migration is a legitimate research finding; it does not block the base migration system.

## Verification Commands and Acceptance Record

These are implementation-time commands; no new feature flags below are claimed to exist. Paths/endpoints in angle brackets must be replaced with real inputs.

```sh
cmake --build build-pp --target test-llama-archs test-save-load-state
./build-pp/bin/test-llama-archs --pp-replicas
./build-pp/bin/test-llama-archs --pp-model <qwen35moe.gguf>
./build-pp/bin/test-save-load-state -m <hybrid.gguf> -ngl 999 --pp-baseline <new-output.json>
```

For RPC, use a separate build and two explicitly configured endpoints; the current build-pp has RPC disabled:

```sh
cmake -S . -B build-pp-rpc -DGGML_CUDA=ON -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-pp-rpc --target test-llama-archs
./build-pp-rpc/bin/test-llama-archs --pp-rpc <endpoint-a> <endpoint-b>
```

The existing `--pp-rpc` fixture and local `--pp-model` path are distinct; do not claim current real-model cross-machine coverage by combining flags. Extend existing harness paths for that acceptance. A test return of 77 denotes unavailable hardware/backend and is not a pass.

For every stage, run the focused existing binary case before/after its change, inspect assertions and status, and record unavailable cases. Add meaningful cases to existing files rather than introducing new test source files. No automatic commits.

Compare: no migration; current full-tensor synchronous reference; optimized live-state copy; asynchronous checkpoint+delta COPY; checkpoint+REPLAY; adaptive COPY/REPLAY; optional summary composition; optional short shadow mode. Include the optimized copy baseline so savings from avoiding unused capacity are not attributed to reconstruction.

Measure total physical link bytes over the whole request, startup/preload bytes separately, logical state bytes, replay and snapshot compute, capture overhead, checkpoint age, lag, request-to-switch time, cutover time, TTIT p50/p95/p99 and maximum around the switch, post-switch steady speed, memory per category, and task completion/tool-action correctness. Count failed/canceled preparations as overhead.

Sweep context length, moved-layer count/mix, both directions, checkpoint age, replay chunk length, available memory, bandwidth/latency and tool idle duration. Compare replay kernels under single-token and chunked execution; runtime fusion paths can differ. Include an overloaded case in which migration is postponed safely.

Use fixed accepted tokens for state/logit diagnosis, then matched end-to-end agent tasks for quality. Native precision is the first baseline; same-seed sampled text is diagnostic, not a proof of equality. Publish quality differences with uncertainty and a predeclared application tolerance rather than treating arbitrary top-k agreement as semantic equivalence.

## First Deliverable and Research Claim

The first end-to-end deliverable is local two-device, one-sequence migration of one configured boundary interval: a checkpoint, a retained input suffix, isolated replay, and a prepared safe switch. Support one full-attention layer first, then linear and mixed intervals, then reverse moves. Complete the nonblocking and timeline gates before calling it low-stall migration.

The research hypothesis is that application idle opportunities and observed PP slack can reduce total state communication by choosing among copying and dependency-correct reconstruction of warm replicas. HYPIC-inspired composition is one optional mechanism within that system. Neither universal zero latency nor lower communication for every GQA interval is promised.

Planning checks completed: baseline and existing interfaces inspected; current build/test registration inspected; RPC serialization and graph lifetime constraints verified from source; task dependencies and state-version conventions checked. Inference, numerical equivalence, performance and task-quality gates remain implementation-time work.
