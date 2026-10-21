#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <stdexcept>

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const llama_layer_device_cb & layer_device,
                     bool   independent_layers) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr,
        "",
        layer_device,
        independent_layers
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr,
        layer_device,
        independent_layers
    )),
    independent_layers(independent_layers),
    active_view(std::make_shared<execution_view>(execution_view {
        mem_attn->get_active_metadata(), mem_recr->get_active_metadata(),
        [&]() {
            std::vector<ggml_backend_dev_t> result;
            if (independent_layers) {
                result.resize(hparams.n_layer());
                for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                    result[il] = layer_device(il);
                }
            }
            return result;
        }(),
        0, 0, true })) {}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    return init_batch(active_view, balloc, n_ubatch, embd_all);
}

llama_memory_context_ptr llama_memory_hybrid::init_batch(
        const execution_view_ptr & view,
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(*view->recurrent, ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(*view->attn, ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, view, std::move(heads_attn), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return init_full(active_view);
}

llama_memory_context_ptr llama_memory_hybrid::init_full(const execution_view_ptr & view) {
    return std::make_unique<llama_memory_hybrid_context>(this, view);
}

llama_memory_context_ptr llama_memory_hybrid::init_ubatches(
        const execution_view_ptr & view,
        std::vector<llama_ubatch> ubatches) {
    if (!view || !mem_recr->prepare(*view->recurrent, ubatches)) {
        return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    auto sinfos = mem_attn->prepare(*view->attn, ubatches);
    if (sinfos.empty()) {
        return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    return std::make_unique<llama_memory_hybrid_context>(this, view, std::move(sinfos), std::move(ubatches));
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    ++active_view->generation;
    mem_attn->clear(data);
    mem_recr->clear(data);
    active_view->frontier = 0;
    active_view->append_only = true;
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    active_view->append_only = false;
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    active_view->append_only = false;
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
    active_view->append_only = false;
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
    active_view->append_only = false;
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
    active_view->append_only = false;
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
    active_view->append_only = false;
}

bool llama_memory_hybrid::is_append_only() const {
    return !independent_layers || active_view->append_only;
}

llama_memory_hybrid::execution_view_ptr llama_memory_hybrid::get_active_view() const {
    return active_view;
}

llama_memory_hybrid::execution_view_ptr llama_memory_hybrid::clone_active_view() const {
    return clone_active_view(active_view->layer_devices);
}

llama_memory_hybrid::execution_view_ptr llama_memory_hybrid::clone_active_view(const std::vector<ggml_backend_dev_t> & layer_devices) const {
    return std::make_shared<execution_view>(execution_view {
        mem_attn->clone_active_metadata(), mem_recr->clone_active_metadata(),
        layer_devices, active_view->frontier, active_view->generation + 1, active_view->append_only });
}

void llama_memory_hybrid::exchange_active_view(const execution_view_ptr & candidate, uint64_t expected_frontier) {
    if (!candidate || candidate->frontier != expected_frontier || active_view->frontier != expected_frontier) {
        throw std::runtime_error("stale hybrid execution view");
    }

    std::swap(active_view->attn->v_heads, candidate->attn->v_heads);
    std::swap(*active_view->attn->v_cells, *candidate->attn->v_cells);
    std::swap(active_view->attn->seq_to_stream, candidate->attn->seq_to_stream);
    std::swap(active_view->attn->sc_info, candidate->attn->sc_info);

    std::swap(active_view->recurrent->head, candidate->recurrent->head);
    std::swap(active_view->recurrent->used, candidate->recurrent->used);
    std::swap(active_view->recurrent->n, candidate->recurrent->n);
    std::swap(active_view->recurrent->rs_z, candidate->recurrent->rs_z);
    std::swap(active_view->recurrent->rs_idx, candidate->recurrent->rs_idx);
    std::swap(active_view->recurrent->cells, candidate->recurrent->cells);

    std::swap(active_view->layer_devices, candidate->layer_devices);
    std::swap(active_view->frontier, candidate->frontier);
    std::swap(active_view->generation, candidate->generation);
    std::swap(active_view->append_only, candidate->append_only);
}

std::vector<ggml_tensor *> llama_memory_hybrid::layer_state_tensors(int32_t il) const {
    if (il < 0 || (uint32_t) il >= hparams.n_layer()) {
        return {};
    }
    return hparams.is_recr(il) ? mem_recr->layer_state_tensors(il) : mem_attn->layer_state_tensors(il);
}

std::vector<ggml_tensor *> llama_memory_hybrid::layer_state_tensors(const execution_view_ptr & view, int32_t il) const {
    if (!view || il < 0 || (uint32_t) il >= hparams.n_layer()) {
        return {};
    }
    auto * dev = view->layer_devices.empty() ? nullptr : view->layer_devices.at(il);
    return hparams.is_recr(il) ? mem_recr->layer_state_tensors(il, dev) : mem_attn->layer_state_tensors(il, dev);
}

llama_memory_layer_move_ptr llama_memory_hybrid::prepare_layer_move(int32_t il, ggml_backend_dev_t dev) {
    if (il < 0 || (uint32_t) il >= hparams.n_layer() || !dev) {
        throw std::invalid_argument("invalid hybrid cache layer destination");
    }
    return hparams.is_recr(il) ? mem_recr->prepare_layer_move(il, dev) : mem_attn->prepare_layer_move(il, dev);
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    view(mem->get_active_view()),
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), view->attn, view->layer_devices)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), view->recurrent, view->layer_devices)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
        llama_memory_hybrid::execution_view_ptr view) :
    view(std::move(view)),
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), this->view->attn, this->view->layer_devices)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->view->recurrent, this->view->layer_devices)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
    GGML_ASSERT(this->view);
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    view(mem->get_active_view()),
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    view(mem->get_active_view()),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), view->attn, std::move(sinfos_attn), this->ubatches, view->layer_devices)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), view->recurrent, this->ubatches, view->layer_devices)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
              llama_memory_hybrid::execution_view_ptr view,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    view(std::move(view)),
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), this->view->attn, std::move(sinfos_attn), this->ubatches, this->view->layer_devices)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->view->recurrent, this->ubatches, this->view->layer_devices)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
    GGML_ASSERT(this->view);
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    if (res && !ubatches.empty()) {
        const auto & ubatch = ubatches[i_next];
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            view->frontier = std::max<uint64_t>(view->frontier, uint64_t(ubatch.pos[i]) + 1);
        }
    }

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
