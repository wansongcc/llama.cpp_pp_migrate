# PP Migration S1: State Bank and Metadata Ownership Design

Status: implemented and verified on the local two-T4 test setup.

The confirmed implementation keeps one stable active metadata object for legacy active-only APIs. Candidate contexts receive their cloned metadata explicitly. Promotion exchanges the contents of the active and candidate metadata objects at an exact frontier, so the legacy references remain stable without sharing mutable candidate state.

The fixed bank implementation allocates each `(layer, device)` state bank once and reuses its tensor object, backend buffer, and data address across repeated A/B/A migration. S1 remains synchronous: bank selection is global during promotion, and concurrent candidate execution is deferred to the per-runtime bank binding in S2.

## Objective

S1 makes state storage addresses stable and prevents candidate work from mutating foreground inference state. It remains synchronous and does not yet run shadow compute, asynchronous transfer, replay, or CUDA Graph warmup.

The central invariant is:

> An execution view may mutate only its own KV and recurrent metadata, and may write only the state banks bound to that view.

The current migration code satisfies neither part for concurrent work. It swaps newly allocated layer tensors into one shared memory object, while KV slot metadata and recurrent cursors remain component-wide mutable fields.

## Proposed objects

### 1. Fixed layer state bank

One bank is identified by `(layer, device, bank generation)` and owns stable backend allocations for that layer:

- Full-attention layer: K and V tensors, including stream views.
- Recurrent layer: R, S, and optional P tensors.
- Device identity and byte capacity.
- State generation and token frontier.
- Lifecycle state: `empty`, `copying`, `ready`, or `active`.
- A completion handle used by later asynchronous phases.

The bank owns tensor storage. Graphs and schedulers only borrow tensor pointers. A bank cannot be recycled while a runtime or in-flight operation still references it.

The experimental path will allocate a bounded set of banks at context construction for the configured active placement and validated adjacent candidate placements. Promotion must not allocate or replace tensor storage.

### 2. Execution metadata view

Metadata is per execution view, not per layer and not global to all candidates. A view contains:

- KV cell contents and sequence membership.
- KV search heads, sequence-to-stream mapping, and pending stream-copy state.
- Recurrent cells, `head`, `used`, `n`, `rs_z`, and rollback indices.
- The token frontier represented by the metadata.
- One selected state bank for every model layer.

The metadata is global across layers because one KV slot layout and one recurrent cell layout are selected for an entire ubatch. Duplicating only the moved layer metadata would allow the active and candidate executions to disagree about slot placement.

The active view and each candidate view have separate mutable metadata objects. Cloning a view is a host-side metadata copy plus a state-bank checkpoint. Candidate graph construction and candidate decode must receive the candidate view explicitly and must not read mutable fields through the active memory object.

The current fields map to the proposed views as follows:

| Current owner | Mutable fields moved into a view | Physical objects kept in banks or shared definitions |
| --- | --- | --- |
| `llama_kv_cache` | `v_heads`, `v_cells_impl`/`v_cells`, `seq_to_stream`, `sc_info` | K/V layer tensors, stream tensor views, immutable cache geometry and types |
| `llama_memory_recurrent` | `head`, `used`, `n`, `rs_z`, `rs_idx`, `cells` | R/S/P layer tensors, immutable sizes and model parameters |
| `llama_memory_hybrid` | append-only status for that execution lineage | references to the attention and recurrent definitions |
| memory batch contexts | selected slot information and references to one view | no ownership of state-bank allocation |

`v_cells` is currently a reference bound to `v_cells_impl`, so it cannot be redirected safely by assigning a new shared pointer. The refactor must remove that permanent reference or move both ownership and access behind the selected view. The recurrent context also reads and mutates its parent memory object directly, including resetting rollback indices in `s_copy()`. Both access patterns must be removed before foreground and candidate execution can coexist.

### 3. Placement slot

A placement slot owns:

- A placement vector.
- Its execution metadata view.
- Its layer-bank bindings.
- A monotonically increasing generation.
- A readiness frontier.

S1 has one active slot and at most one prepared candidate slot. Later phases can add a candidate scheduler, activation journal, stream, events, and CUDA Graph state without changing bank ownership.

The slot does not own weights. `active_layers` continues to select model-owned immutable replicas. It also does not own a scheduler in S1; scheduler ownership is intentionally deferred to S2.

## S1 operations

### Context initialization

1. Construct the active banks and active metadata view.
2. Preallocate destination banks for validated adjacent placement changes.
3. Keep all tensor addresses stable for the context lifetime.

### Prepare candidate at frontier P

1. Require a safe token boundary and no in-flight graph.
2. Clone active KV and recurrent metadata into the candidate view.
3. Copy affected layer tensors from the active banks to fixed destination banks.
4. Bind unchanged layers to their existing banks as read-only inputs for the candidate slot.
5. Mark all copied banks and the candidate view ready at frontier P.

S1 preparation is synchronous. The foreground cannot advance between the metadata snapshot and completion. S3 will relax this with events and frontier tracking.

### Promote candidate

1. Verify placement, generation, bank readiness, and an exact frontier match.
2. Select the candidate slot as active.
3. Reset or select scheduler state for that placement.
4. Preserve the previous active slot until rollback is no longer possible.

Promotion changes handles and placement identity only. It performs no tensor allocation or state copy. S1 can still pay scheduler reserve cost because the persistent candidate runtime is introduced in S2.

### Abort or rollback

Failure before promotion discards candidate readiness without touching the active slot. Failure after slot selection restores the previous slot and placement generation. A bank is reusable only after all scheduler and backend references have been released.

## Required refactoring boundaries

The minimum safe refactor is:

1. Extract KV mutable metadata from `llama_kv_cache` into an explicit view object.
2. Extract recurrent mutable metadata from `llama_memory_recurrent` into an explicit view object.
3. Make `llama_kv_cache_context` and `llama_memory_recurrent_context` hold the selected view rather than reach through one global memory object.
4. Replace temporary `layer_move` allocation ownership with persistent bank lookup and bank-to-bank copy.
5. Make `llama_memory_hybrid_context` bind one coherent hybrid execution view.

The proposed minimum internal interface is behavioral rather than a new public API:

- Create a metadata view from the active view at a safe frontier.
- Look up or preallocate a layer bank for a validated device.
- Copy one layer from a source bank to a destination bank without publishing it.
- Bind a complete execution view while constructing a memory batch context.
- Publish a prepared placement slot only after frontier and generation validation.
- Retire or retain the previous slot without freeing storage referenced by a graph.

The existing synchronous `migrate_layers_sync()` remains the external reference operation. During S1 it will be reimplemented as prepare plus immediate promote, followed by the existing reserve path. This keeps the behavior reviewable before S2 changes scheduler ownership.

This does not use state serialization as an internal cloning mechanism. Serialization includes compatibility and layout work that is unnecessary for an in-process checkpoint and would hide transfer costs.

## Explicit non-goals for S1

- No concurrent foreground and candidate execution.
- No asynchronous CUDA copy.
- No activation journal or replay.
- No layer shadow compute.
- No candidate scheduler or CUDA Graph capture.
- No KV live-row compression.

These remain separate phases so metadata isolation can be validated before concurrency is introduced.

## Validation gates

S1 is complete only if the existing test executable verifies:

- Active decode bytes and logits do not change while a candidate is prepared.
- Candidate metadata can be mutated in a test probe without changing active metadata.
- Prepare, promote, reverse prepare, and rollback preserve exact state bytes and logits.
- Repeated A/B/A promotion reuses the same tensor addresses and does not increase backend memory.
- A stale candidate frontier is rejected.
- An injected preparation or promotion failure leaves the active view usable.
- Full-attention, recurrent, and mixed intervals all pass on two devices.

Address stability is checked using tensor object address, `tensor->data`, backend buffer identity, and total device memory. Checking only one of these is insufficient because a tensor object can survive while its allocation changes, or a new tensor object can view the same buffer with a different graph identity.

## Confirmed decision

> Use one complete mutable metadata view per active or candidate placement, while keeping fixed per-layer, per-device tensor banks. Do not attempt per-layer metadata duplication.

This costs an additional host-side KV cell table and recurrent cell table per candidate. It avoids shared cursor mutation and matches the unit at which ubatches allocate KV and recurrent slots.
