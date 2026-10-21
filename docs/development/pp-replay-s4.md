# S4: checkpoint, owned hidden journal, and interval replay

This is the implementation and completed acceptance record for the local interval-replay stage. Raw logs and tested source/binary fingerprints are archived under `artifacts/pp-replay-s4/`.

## Execution contract

A frontier P means that tokens [0, P) have been processed. A migration request first copies the selected layers' checkpoint at P into their fixed destination banks. The foreground continues to F. The journal stores the pre-norm residual input to the first moved layer for each token in [P, F).

Replay executes only the selected interval [first, end), in the original layer order. It reads the checkpoint, original positions and token metadata, and owned hidden inputs. It writes the destination banks and the candidate metadata view. It does not run the remaining model layers, the final output normalization, the LM head, or a sampler.

After all replay work completes, the candidate frontier is published. Commit still requires matching source runtime generation, source view generation, append-only state, valid journal, and equal active/candidate frontiers. The existing full-placement candidate runtime takes over subsequent ordinary decoding.

The foreground is paused during the explicit replay call. This stage does not implement PP-bubble scheduling or concurrent catch-up.

## Interfaces and ownership

The experimental C++ interface in `llama-context.h` adds:

```cpp
bool enable_layer_migration_replay(uint64_t request_id, size_t capacity_tokens);
bool replay_layer_migration(uint64_t request_id, int32_t fail_after_tokens = -1);
```

The failure argument is for regression injection: -1 disables injection, 0 fails before computation, and a positive value fails after that many tokens have completed.

Typical use:

```cpp
const auto id = ctx.migrate_layers_async_begin(first, end, destination);
if (!ctx.enable_layer_migration_replay(id, capacity_tokens)) {
    ctx.migrate_layers_async_cancel(id);
    return;
}

// Ordinary single-token decode calls record the missing inputs.

if (!ctx.replay_layer_migration(id)) {
    ctx.migrate_layers_async_cancel(id);
    return;
}
if (!ctx.migrate_layers_async_commit(id)) {
    // A rejected commit has already discarded the pending request.
    return;
}
```

One pending migration owns one bounded journal and at most one replay runtime. Entries own contiguous F32 hidden values, original position components, the original token ID when present, and the input-kind flag. The request owns the sequence/runtime identity and interval, so these fields need not be repeated in each entry.

Capture explicitly materializes a separate graph output with `ggml_dup` at the interval entry. After foreground completion, it copies that output into host-owned journal storage. Capture uses synchronization; there are no unfinished journal-copy events or borrowed scratch pointers to pin. An entry remains owned until a replay call completes successfully or the request is discarded.

Successful replay clears the consumed entries, allowing repeated catch-up with a capacity of one token. Overflow invalidates the candidate and preserves foreground decoding. Exceptions during capture also invalidate the candidate instead of aborting foreground decoding.

The replay scheduler and graph result are independent of the full candidate TG scheduler. Replay never replaces the TG graph used after promotion. Replay graph validation checks that all selected layers are present, that no out-of-range layer is present, that layer operations execute on their assigned backends, and that there is no logits output.

## Model and graph changes

Qwen3Next and Qwen3.5-MoE reuse their existing layer loops and attention/FFN helpers. Graph parameters add the interval bounds and capture layer to the graph-reuse identity. Normal inference retains the full layer range.

A dedicated hidden input supplies the residual directly. It deliberately does not use the token-embedding lookup, even when the replay batch retains original token IDs for memory metadata.

This distinction matters for Qwen3.5 M-RoPE: replaying text as a generic embedding batch produced correct recurrent tensors but different serialized KV metadata. Keeping the original token/input-kind and positions fixes this discrepancy. No numerical tolerance was relaxed to accept it.

Hybrid graph inputs skip buffers absent from a partial graph. A recurrent-only interval does not build position inputs that would be unused by that graph.

Clearing hybrid memory advances the view generation. An old checkpoint cannot become valid again merely because a new sequence reaches the same numerical frontier.

## Supported scope

- Qwen3Next and Qwen3.5-MoE with the existing experimental validated placements.
- A continuous layer interval, moved across an adjacent PP boundary while preserving device order.
- One append-only sequence, sequence ID 0, with single-token foreground decode.
- F32 entry hidden, original state precision, and the normal TG attention kernels.
- Full-attention, recurrent, and mixed intervals, in both supported migration directions.
- Empty replay, repeated catch-up, cancellation, and failure before or after candidate writes.

Multi-token foreground chunks invalidate the journal. Chunk GDN and token GDN are different execution paths and require a separate numerical acceptance study. Multi-sequence replay, adapter changes, context shifting, arbitrary PP reorderings, activation quantization, and RPC performance are outside this acceptance stage.

The acceptance configuration uses F16 KV and F32 recurrent state. Quantized KV formats are not certified by this test matrix.

The checkpoint path still copies allocated state tensors. The valid-row copy implemented in the test is a comparison benchmark, not a new production migration transport.

## Measurement definitions

- `journal_bytes`: cumulative F32 hidden payload captured by the request.
- `journal_peak_bytes`: peak retained hidden/position/token payload; excludes STL allocator overhead and shared request metadata.
- `replay_compute_bytes`: peak scheduler compute-buffer bytes across backends; excludes model weights, state banks, and CUDA executable caches.
- `replay_us`: wall time of explicit catch-up, including initial replay-runtime setup and graph work.
- `replay_step_us`: completed token-step timings, capped at 1024 samples per request.
- `shadow_shared_bytes`: host read plus restore bytes for the old full-model shadow reference path.
- `shadow_us`: full-model shadow wall time, including shared-state snapshots and restoration.

The pre-existing `total_us` measures request lifetime from begin through commit, including foreground work between those calls. It is not a pause measurement. The pre-existing asynchronous `copy_us` measures enqueue time; the valid-copy benchmark below uses completed synchronous copies instead.

All byte measurements are local tensor payload or host-transfer accounting. They are not network wire measurements. Checkpoint initialization must be included when reporting lifecycle communication; a pre-existing checkpoint can be excluded only from the subsequent gap comparison.

A one-shot direct migration may avoid the earlier checkpoint entirely. Its total payload must therefore also be compared against checkpoint plus hidden, not only against the gap-copy payload. Preinstalled weight replicas are shared setup assumptions for all three paths; these measurements do not include their initial distribution.

The copy reference sends missing valid KV rows [P, F) and the full recurrent state at F using synchronous backend tensor copies. It restores the saved candidate checkpoint before interval replay. It measures 100 copies after 10 warm-up copies. Transposed V uses per-column spans; this reference is not a packed network transport.

Separate token-gap measurements place no state serialization between the last foreground token, replay/commit, and the next output token. Logs containing `catchup+commit+validation` include test state reads and must not be presented as application migration latency.

## Reproduction

```sh
cmake --build build-pp-cuda --target test-llama-archs -j 8

CUDA_VISIBLE_DEVICES=0,1 build-pp-cuda/bin/test-llama-archs \
    --pp-replicas --seed 627113891

CUDA_VISIBLE_DEVICES=0,1 build-pp-cuda/bin/test-llama-archs \
    --pp-model Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
    --pp-low-vram --pp-replay-acceptance --seed 1234

CUDA_VISIBLE_DEVICES=0,1 build-pp-cuda/bin/test-llama-archs \
    --pp-model Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
    --pp-low-vram --pp-replay-costs --seed 1234
```

The low-VRAM configuration includes a CPU prefix and two GPU stages. Its foreground latency is not representative of an all-GPU deployment. Do not run CPU-heavy comparisons concurrently when collecting latency results.

The small-model matrix covers P = 0, 6, 255; gaps 0, 1, 8, 32, 128; three interval types; two directions; and 128 continuation tokens. It also performs 100 round trips and checks fixed bank addresses and sizes.

The real-generation test uses 20 fixed text inputs, 10 greedy and 10 fixed-seed stochastic runs, and up to 128 generated tokens per input, stopping only on EOG. It requires byte-exact serialized state, zero logit error, and identical sampled tokens. These are finite regression results, not a proof of equivalence for every possible input or backend.

## Acceptance results

The final small-model run, cost comparison, and frozen-binary real-generation rerun all passed. Source and executable SHA-256 checks match the archived manifest. The final generation run reached its success summary and the process is no longer running; its original shell session handle expired across the interruption, so a recovered shell exit status is not claimed for that run.

Raw logs are under `artifacts/pp-replay-s4/`. `source-and-binary.sha256` identifies the tested source files and executables; `model.sha256` identifies the GGUF. `hardware.txt` records the devices and driver. The repository ignores `*.log`, so these raw logs are workspace artifacts rather than automatically tracked source files.

### Correctness and isolation

| Check | Result | Evidence |
| --- | --- | --- |
| P = 0, 6, 255; D = 0, 1, 8, 32, 128; 3 interval types; 2 directions | 90 cases passed; byte-exact sequence state and 128 exact continuation logits per case | `matrix.log` |
| Repeated migration | 100 round trips; bank addresses and sizes stable; logits exact | `matrix.log` |
| Bounded consumption | 5 replay segments with journal capacity 1 | `matrix.log` |
| Overflow, pre-compute failure, failure after a completed replay token, cancel, post-switch injected failure | Foreground state, placement, and continued logits unchanged | `matrix.log` |
| Missing hidden prefix and old sequence epoch | Rejected; foreground continuation exact | `matrix.log` |
| Multi-token chunk input | Rejected by token replay; chunk foreground and continuation preserved | `matrix.log` |
| Real generation | 20 workloads, 2560 tokens, 10 greedy and 10 seeded runs; byte-exact state, zero logit error, identical sampled tokens | `real-generation.log` |

Every real workload produced 128 tokens without early EOG. The last workload rebuilt a 161-token gap at P = 1. The final generation run recorded 260 journal-maintenance samples: baseline foreground p50/p95 140.314/147.090 ms and journal foreground p50/p95 136.052/153.482 ms. These are run-specific observations, not a zero-overhead claim.

The final small-model stability run recorded 190 post-warmup migration samples. Replay p50/p95 was 1.415/1.864 ms, and handle-switch p50/p95 was 0.000/0.001 ms at the timer's microsecond resolution. The latter is not end-to-end migration latency. Peak retained journal payload was 1032 bytes and replay compute buffers were 291456 bytes in this capacity-one fixture.

### Real-model costs

Configuration: Qwen3.5-35B-A3B Q4_K_M, CPU prefix plus two Tesla T4 stages, context capacity 512, P = 6, D = 100. These are local synchronous measurements, not network or overlap results. Each valid-copy measurement contains 100 repetitions after 10 warmup repetitions. Replay and shadow each process 100 sequential missing tokens.

| Interval | Valid gap-copy payload | Hidden payload | Valid copy p50/p95, ms | Whole interval replay, ms | Whole-model shadow, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| recurrent [36, 37) | 2195456 B | 819200 B | 0.540 / 0.542 | 84.636 | 20641.214 |
| full [35, 36) | 204800 B | 819200 B | 0.101 / 0.103 | 78.651 | 20199.611 |
| mixed [35, 37) | 2400256 B | 819200 B | 0.640 / 0.646 | 135.564 | 19832.528 |

The hidden payload excludes per-token position/token metadata (another 2000 B for these 100-token journals), shared request metadata, and any future network framing. The payload reduction for an already-present checkpoint is 62.69% for recurrent and 65.87% for mixed. Full attention requires four times as much hidden payload as missing KV payload in this configuration.

Copying is much faster than replay on this local link. Replay saves communication for some intervals at an additional compute cost; it does not dominate direct copying.

| Interval | Baseline foreground p50/p95, ms | Journal foreground p50/p95, ms | Full-shadow foreground p50/p95, ms |
| --- | ---: | ---: | ---: |
| recurrent | 140.252 / 146.179 | 138.934 / 142.920 | 350.127 / 376.037 |
| full | 139.452 / 145.915 | 139.950 / 145.893 | 339.754 / 371.295 |
| mixed | 139.013 / 142.772 | 152.876 / 155.670 | 338.860 / 349.484 |

The mixed journal run has a 9.97% p50 increase. Small negative differences in the other runs are not evidence that logging accelerates inference. These single-run measurements do not establish a latency SLA.

The separate reverse-direction D = 1 token-gap probes measured 168.985, 178.337, and 171.426 ms for recurrent/full/mixed replay, respectively. They include catch-up, commit, and the next decode, with checkpoint preparation completed earlier. These are single probes, not p95 statistics. The corresponding already-tracked full-shadow probes measured 148.253, 141.310, and 144.023 ms.

The prototype creates the replay runtime when catch-up first runs. Moving this preparation into an earlier idle window, retaining suitable replay graphs, scheduling bounded work in real PP bubbles, and proving native TG readiness remain later optimization work.

### Lifecycle communication

The production checkpoint still copies allocated tensors. For this P = 6, F = 106 experiment:

| Interval | One-shot direct valid state at F | Actual checkpoint at P plus hidden | Ideal valid-only checkpoint at P plus hidden |
| --- | ---: | ---: | ---: |
| recurrent | 2195456 B | 3014656 B | 3014656 B |
| full | 217088 B | 1867776 B | 831488 B |
| mixed | 2412544 B | 4063232 B | 3026944 B |

Thus this experiment does not show a total-traffic win over a one-shot direct migration that never established an earlier checkpoint. It shows smaller gap traffic for recurrent/mixed intervals when a checkpoint is already present, and a much cheaper reconstruction engine than full-model shadow. Checkpoint reuse, placement policy, and slow-link measurements remain necessary for the broader research claim.

Failed/cancelled fixture requests also report their discarded checkpoint and hidden payloads in `matrix.log`. These costs are not silently omitted from the failure tests.
