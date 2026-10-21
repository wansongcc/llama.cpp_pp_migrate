# Synchronous PP migration: phase-one measurements

This document records the phase-one reference result for synchronous layer migration. The test is diagnostic infrastructure, not a production performance claim.

## Measurement definitions

- State-copy bytes are the exact tensor bytes passed to `ggml_backend_tensor_copy`. They include the full allocated layer-state buffers, not only rows used by the current token prefix.
- State-copy bytes do not include transport framing or staging overhead. A local CUDA copy and an RPC transfer can therefore have the same state-copy byte count but different physical link traffic.
- Migration blocking time is the time spent inside `llama_context::migrate_layers_sync` before the caller can invoke decode again.
- Trigger-to-next-token time is migration blocking time plus the first post-migration single-token decode.
- Per-device layer time uses the existing scheduler evaluation callback. It synchronizes at every `l_out` node to isolate layer execution. This changes normal pipeline overlap, so the result is a compute-side diagnostic and must not be reported as production decode latency.

## Test configuration

- Date: 2026-10-17
- GPUs: 2x NVIDIA Tesla T4
- CUDA architecture: 7.5
- Model: generated Qwen35MoE test fixture, F32, 4 layers
- Attention pattern: recurrent, full, recurrent, full
- Initial PP boundary: layer 3
- Context size: 64
- Batch and micro-batch size: 1
- Migration trigger: after positions 0 through 4, before decoding position 5
- Seed: 1234
- Moved interval for the mixed case: layers `[1, 3)`

The Qwen35MoE fixture uses four layers so that a full-attention layer followed by a recurrent layer is adjacent to a PP boundary. This keeps the test small while covering KV-only, recurrent-only, and mixed migration.

## Results

| Case | State copied | Repeated forward migration, median | Blocking at measured trigger | First post-migration decode | Trigger to next token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recurrent layer `[2, 3)` | 140288 bytes | 3.879 ms | 5.002 ms | 2.463 ms | 7.473 ms |
| Full-attention layer `[1, 2)` | 262144 bytes | 3.973 ms | 6.467 ms | 1.891 ms | 8.368 ms |
| Mixed layers `[1, 3)` | 402432 bytes | 4.110 ms | 3.651 ms | 1.929 ms | 5.593 ms |

The single measured trigger can differ from the repeated-migration median because scheduler reserve time varies. For the mixed case, the measured blocking phases were:

| Phase | Time |
| --- | ---: |
| Synchronize queued work | 0.003 ms |
| Copy layer state | 0.172 ms |
| Other destination preparation | 0.027 ms |
| Switch state and placement | 0.009 ms |
| Rebuild and reserve scheduler graphs | 3.439 ms |
| Total | 3.651 ms |

Scheduler reserve accounts for most of the synchronous pause in this fixture. State transfer is not the dominant local-GPU cost at this context size.

The compute probe moved layers `[1, 3)` from CUDA0 to CUDA1 and collected four tokens before and four tokens after migration:

| Device-side observation | Before | After |
| --- | ---: | ---: |
| CUDA0 layer time, median | 0.976 ms | 0.333 ms |
| CUDA1 layer time, median | 0.270 ms | 0.825 ms |
| Moved-layer time | 0.608 ms on CUDA0 | 0.560 ms on CUDA1 |

CUDA0 layer time decreased by 65.9 percent in this diagnostic mode. The moved-layer work disappeared from CUDA0 and appeared on CUDA1. This confirms the expected compute shift on one card. The percentage must not be extrapolated to a production model because the fixture is very small and the probe disables layer overlap.

All three migration cases produced exact logits for this fixture: maximum absolute and relative differences were both zero. Repeated forward and reverse moves, injected rollback, state-byte equality, layer ownership, and state allocation ownership also passed.

## Reproduction

```sh
cmake --build build-pp-cuda --target test-llama-archs -j 8
mkdir -p /tmp/llama-pp-phase1-models
build-pp-cuda/bin/test-llama-archs -a qwen35moe -o /tmp/llama-pp-phase1-models -s 1234 -v 0
CUDA_VISIBLE_DEVICES=0,1 build-pp-cuda/bin/test-llama-archs --pp-model /tmp/llama-pp-phase1-models/qwen35moe-moe.gguf -s 1234
CUDA_VISIBLE_DEVICES=0,1 build-pp-cuda/bin/test-llama-archs --pp-replicas -s 1234
```

## Interpretation and next measurement

Phase one establishes an exact local-GPU baseline and exposes the dominant pause. The next benchmark should use a real target model and its production context length. It should repeat the same counters over RPC or the intended edge link, then add physical transport-byte counters because tensor bytes alone do not include protocol overhead.
